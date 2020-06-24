from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ftl

def load_image(image_path):
    """load_image Load an image in a thread
    
    This method loads and image and converts the image
    to boolean values, where each value indicates whether
    the pixel is non-zero.

    Args:
        image_path (pathlib.Path): Path to file to load

    Returns:
        np.ndarray: A boolean n-dimensional array
    """
    # Attach the jvm to the thread
    jutil.attach()
    
    br = BioReader(str(image_path.absolute()))
    image = np.squeeze(br.read_image())>0
    
    # Detach the jvm
    jutil.detach()
    
    return br,image
    
def save_image(image_path,br,image):
    """save_image Save an image in a thread

    Args:
        image_path (pathlib.Path): Path to save image to (including name)
        br (bfio.BioReader): BioReader object used to load the image
        image (np.ndarray): ndarray with at most three dimensions (XYZ)

    Returns:
        string: Name of the image that was saved
    """
    # Attach the jvm to the thread
    jutil.attach()
    
    bw = BioWriter(str(image_path.absolute()),metadata=br.read_metadata())
    bw.write_image(np.reshape(image,(br.num_y(),br.num_x(),br.num_z(),1,1)))
    bw.close_image()
    
    del br
    del bw
    
    # Detach the jvm
    jutil.detach()
    
    return image_path.name

if __name__ == '__main__':
    
    # Setup the argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Label objects in a 2d or 3d binary image.')
    parser.add_argument('--connectivity', dest='connectivity', type=str,
                        help='City block connectivity, must be less than or equal to the number of dimensions', required=True)
    parser.add_argument('--images', dest='images', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--process', dest='process', type=str,
                        help='Process number', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    connectivity = int(args.connectivity)
    images = args.images.split(',')
    outDir = args.outDir
    process = args.process
    
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("process {}".format(process))
    logger.setLevel(logging.INFO)
    
    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    
    # Start the threadpool
    logger.info('Starting ThreadPoolExecutor...')
    thread_pool = ThreadPoolExecutor(max_workers=2)
    
    try:
        # Start reading the first image
        f = Path(images[0])
        logger.info('Starting load thread for {}'.format(f.name))
        file_path = str(f.absolute())
        load_thread = thread_pool.submit(load_image,f)
        save_thread = None
        
        for i,image in enumerate(images):
            # Wait for the image to load
            br,bool_image = load_thread.result()
            logger.info('Finished loading image {}'.format(f.name))
            
            # If more image can be loaded, start loading the next image
            if i < len(images)-1:
                load_thread = thread_pool.submit(load_image,Path(images[i+1]))

            # Label the image
            start = time.time()
            out_image = ftl.label_nd(bool_image,connectivity)
            logger.info('Finished labeling in {}s!'.format(time.time() - start))

            # Wait for the prior write thread to finish if it exists
            if save_thread != None:
                image_name = save_thread.result()
                logger.info('Finished writing image {}'.format(image_name))
            
            # Start saving thread for the current image
            f = Path(outDir).joinpath(f.name)
            logger.info('Start writing image {}'.format(str(f.name)))
            save_thread = thread_pool.submit(save_image,f,br,out_image)
            
            # Get the next image directory
            if i < len(images)-1:
                f = Path(images[i+1]).absolute()
        
        # Wait for the final save thread to finish
        image_name = save_thread.result()
            
    finally:
        # Shutdown the jvm
        logger.info('Closing the javabridge...')
        jutil.kill_vm()
        
        # Close the thread pool
        logger.info('Closing the ThreadPoolExecutor...')
        
        thread_pool.shutdown()