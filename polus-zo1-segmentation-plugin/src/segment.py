from bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, os
import numpy as np
from pathlib import Path

# Set the environment variable to prevent odd warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import tensorflow as tf

class ReflectionPadding2D(keras.layers.Layer):
    """ReflectionPadding2D Custom class to handle matconvnet padding
    
    This class is a Keras layer that does reflection padding, which
    is the default method of padding in matconvnet pooling operations.

    Modified from the following:
    https://stackoverflow.com/a/60116269
    
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        if s[1] == None:
            return (None, None, None, s[3])
        return (s[0],
                s[1] + self.padding[0][0] + self.padding[0][1],
                s[2] + self.padding[1][0] + self.padding[1][1],
                s[3])

    def call(self, x, mask=None):
        (top_pad, bottom_pad), (left_pad, right_pad) = self.padding
        return tf.pad(x, [[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        return config
    
def imboxfilt(image,window_size):
    """imboxfilt Use a box filter on a stack of images

    This method applies a box filter to an image. The input is assumed
    to be a 4D array, and should be pre-padded. The output will be smaller
    by window_size-1 pixels in both width and height since this filter does
    not pad the input to account for filtering.

    Args:
        image ([numpy.darray]): A 4d array of images
        window_size ([int]): An odd integer window size

    Returns:
        [numpy.array]: A filtered array of images.
    """

    assert isinstance(image,np.ndarray), 'image must be an ndarray'
    assert isinstance(window_size,int), 'window_size must be an integer'
    assert window_size % 2 == 1, 'window_size must be an odd integer'
    
    # Generate an integral image
    image_ii = image.cumsum(1).cumsum(2)
    
    # Create the output
    output = image_ii[:,0:-window_size,0:-window_size,:] + \
             image_ii[:,window_size:,window_size:,:] - \
             image_ii[:,window_size:,0:-window_size,:] - \
             image_ii[:,0:-window_size,window_size:,:]
             
    return output

def local_response(image,window_size):
    """local_response Regional normalization

    This method normalizes each pixel using the mean and standard
    deviation of all pixels within the window_size. The window_size
    paramter should be 2*radius+1 of the desired region of pixels
    to normalize by. The image should be padded by window_size//2
    on each side.

    Args:
        image ([numpy.ndarray]): 4d array of image tiles
        window_size ([int]): Size of region to normalize

    Returns:
        [numpy.ndarray]: 4d array of image tiles
    """
    image = image.astype(np.float64)
    local_mean = imboxfilt(image,window_size)/(window_size ** 2)
    local_mean_square = imboxfilt(image ** 2,window_size)/(window_size ** 2)
    local_std = np.sqrt(local_mean_square - (local_mean ** 2))
    local_std[local_std<10**-3] = 10**-3
    response = (image[:,
                      window_size//2:-window_size//2,
                      window_size//2:-window_size//2,
                      :] - local_mean)/local_std
    
    return response

if __name__=="__main__":

    # Setup the argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Segment epithelial cell borders labeled for ZO1 tight junction protein.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--images', dest='images', type=str,
                        help='Images to segment', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    outDir = args.outDir
    images = args.images
    
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("segment")
    logger.setLevel(logging.INFO)
    
    # Load the neural network
    model = keras.models.load_model(str(Path(__file__).parent.joinpath('cnn')))
    
    # Define preprocessing and neural network params
    window_size = 127
    max_norm = 6
    tile_size = [256+window_size, 256+window_size]
    tile_stride = [200,200]
    
    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    inpDir_files = str(images).split(',')
    
    # Loop through files in inpDir image collection and process
    try:
        for f in inpDir_files:
            # Initialize the reader/writer objects
            logger.info('Segmenting image: {}'.format(f))
            br = BioReader(str(Path(inpDir).joinpath(f)))
            bw = BioWriter(str(Path(outDir).joinpath(f)),metadata=br.read_metadata())
            
            # Initialize the generators
            batch_size = min([20,br.maximum_batch_size(tile_size=tile_size,tile_stride=tile_stride)])
            readerator = br.iterate(tile_size=tile_size,tile_stride=tile_stride,batch_size=batch_size)
            writerator = bw.writerate(tile_size=tile_size,tile_stride=tile_stride,batch_size=batch_size)
            next(writerator)
            
            batch = 0
            for tiles,indices in readerator:
                logger.info('{}: batch {}'.format(f,batch))
                batch += 1
                
                # Preprocess the image before segmentation
                norm = local_response(tiles,window_size)
                norm[norm>max_norm] = max_norm
                norm[norm<-max_norm] = -max_norm
                
                # Segment the images
                seg = model.predict(norm)
                seg = (seg>0).astype(tiles.dtype)
                
                # Write the tiles to disk
                writerator.send(seg)
            
            # Close the image
            bw.close_image()
    
    finally:
        # Close the javabridge
        logger.info('Closing the javabridge...')
        jutil.kill_vm()