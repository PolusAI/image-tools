from bfio import BioReader, BioWriter,JARS
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
import tifffile
import cv2
import ast
from scipy import ndimage

Tile_Size = 256

def invert_binary(image, kernel=None, n=None):
    invertedimg = np.zeros(image.shape,dtype=datatype)
    invertedimg = 1 - image
    return invertedimg

def dilate_binary(image, kernel=None, n=None): 
    dilatedimg = cv2.dilate(image, kernel, iterations=n)
    return dilatedimg

def erode_binary(image,kernel=None, n=None):
    erodedimg = cv2.erode(image, kernel, iterations=n)
    return erodedimg

def open_binary(image,kernel=None, n=None):
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openimg

def close_binary(image,kernel=None, n=None):
    closeimg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closeimg

def morphgradient_binary(image,kernel=None, n=None):
    mg = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return mg

def skeleton_binary(image,kernel=None, n=None):
    done = False
    size = np.size(image)
    skel = np.zeros(image.shape,datatype)

    while (not done):
        erode = cv2.erode(image,kernel)
        temp = cv2.dilate(erode,kernel)
        temp = cv2.subtract(image,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = erode.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros==size:
            done = True

    skel = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return skel

def holefilling_binary(image,kernel=None, n=None):
    hf = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    return hf

def hitormiss_binary(image,kernel=None, n=None):
    hm = ndimage.binary_hit_or_miss(image, structure1=n)
    return hm

def tophat_binary(image,kernel=None, n=None):
    tophat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
    return tophat

def blackhat_binary(image,kernel=None, n=None):
    blackhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def areafiltering_min_binary(image, kernel=None, n=None):
    min_size = n
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    af_min = np.zeros((image.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            af_min[output == i + 1] = 1

    return af_min

def areafiltering_max_binary(image, kernel=None, n=None):
    max_size = n
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    af_max = np.zeros((image.shape))
    for i in range(0, nb_components):
        if sizes[i] <= max_size:
            af_max[output == i + 1] = 1

    return af_max


if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a WIPP plugin.')

    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--Operation', dest='operations', type=str,
                        help='The types of operations done on image in order', required=True)
    parser.add_argument('--kernelsize', dest='all_kernel', type=int,
                        help='Kernel size that should be used for all operations', required=True)
    parser.add_argument('--structuringshape', dest='struct_shape', type=str,
                        help='Shape of the structuring element can either be Elliptical, Rectangular, or Cross', required=True)

    # Extra arguments based on operation
    parser.add_argument('--MinSize', dest='min_size', type=int,
                        help='Minimum size of pixel that remains', required=False)
    parser.add_argument('--MaxSize', dest='max_size', type=int,
                        help='Maximum size of pixel that remains', required=False)
    parser.add_argument('--iterations', dest='num_iterations', type=int,
                        help='Number of Iterations to apply operation', required=False)
    
    # Input arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    operations = args.operations
    logger.info('Operation = {}'.format(operations))
    intkernel = args.all_kernel
    logger.info('Kernel Size: {}x{}'.format(intkernel, intkernel))

    # structshape = cv2.MORPH_ELLIPSE
    if args.struct_shape == 'Elliptical':
        structshape = cv2.MORPH_ELLIPSE
    elif args.struct_shape == 'Rectangular':
        structshape = cv2.MORPH_RECT
    elif args.struct_shape == 'Cross':
        structshape = cv2.MORPH_CROSS
    else:
        raise ValueError("Structuring Shape is not correct")
    logger.info('Structuring Shape = {}'.format(args.struct_shape))

    min_size = args.min_size
    max_size = args.max_size
    iterations = args.num_iterations

    if 'filter_area_with_min' in operations:
        if min_size == None:
            raise ValueError('Need to specify the minimum of the pixel area to keep')

    if 'filter_area_with_max' in operations:
        if max_size == None:
            raise ValueError('Need to specify the maximum of the pixel area to keep')

    
    if 'dilation' or 'erosion' in operations:
        if iterations == None:
            raise ValueError("Need to specify the number of iterations to apply the operation")

    # A dictionary specifying the function that will be run based on user input. 
    dispatch = {
        'invertion': invert_binary,
        'opening': open_binary,
        'closing': close_binary,
        'morphological_gradient': morphgradient_binary,
        'dilation': dilate_binary,
        'erosion': erode_binary,
        'skeleton': skeleton_binary,
        'fill_holes': holefilling_binary,
        'top_hat': tophat_binary,
        'black_hat': blackhat_binary,
        'filter_area_with_min': areafiltering_min_binary,
        'filter_area_with_max': areafiltering_max_binary
    }

    # Additional arguments for each function
    dict_n_args = {
        'dilation': iterations,
        'erosion': iterations,
        'invertion': None,
        'opening': None,
        'closing': None,
        'morphological_gradient': None,
        'skeleton': None,
        'fill_holes': None,
        'top_hat': None,
        'black_hat': None,
        'filter_area_with_min' : min_size,
        'filter_area_with_max' : max_size
    }
    

    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    # Get all file names in inpDir image collection
    inpDir_files = [f.name for f in Path(inpDir).iterdir()]
    logger.info("Files in input directory: {}".format(inpDir_files))

      
    # Loop through files in inpDir image collection and process
    try:
        imagenum = 0
        for f in inpDir_files:

            # Load an image
            image = Path(inpDir).joinpath(f)

            # Read the image
            br = BioReader(str(image.absolute()))
            # br = BioReader(image.absolute(),max_workers=max([cpu_count()-1,2])) # Version bfio 2.4.4

            # Get the dimensions of the Image
            br_y, br_x, br_z = br.num_y(), br.num_x(), br.num_z()
            datatype = br.read_metadata().image().Pixels.get_PixelType()
            logger.info("Original Datatype {}:".format(datatype))

            # Initialize Kernel
            kernel = cv2.getStructuringElement(structshape,(intkernel,intkernel))

            # Initialize Output
            newfile = Path(outDir).joinpath(f)
            bw = BioWriter(file_path=str(newfile), metadata=br.read_metadata())

            # Initialize the Python Generators to go through each "tile" of the image
            tsize = Tile_Size + (2*intkernel)
            readerator = br.iterate(tile_stride=[Tile_Size, Tile_Size],tile_size=[tsize, tsize], batch_size=1)
            writerator = bw.writerate(tile_size=[Tile_Size, Tile_Size], tile_stride=[Tile_Size, Tile_Size], batch_size=1)
            next(writerator)

            for images,indices in readerator:
                # Extra tiles do not need to be calculated.
                    # Indices should range from -intkernel < index value < Image_Dimension + intkernel
                if indices[0][0][0] == br_x - intkernel:
                    continue
                if indices[1][0][0] == br_y - intkernel:
                    continue

                logger.info(indices)

                # Images are (1, Tile_Size, Tile_Size, 1)
                # Need to convert to (Tile_Size, Tile_Size) to be able to do operation
                images = np.squeeze(images)
                images[images == 255] = 1

                # Initialize which function we are dispatching
                function = dispatch[operations]
                if callable(function):
                    trans_image = function(images, kernel=kernel, n=dict_n_args[operations])
                    trans_image[trans_image==1] = 255

                # The image needs to be converted back to (1, Tile_Size_Tile_Size, 1) to write it
                reshape_img = np.reshape(trans_image[intkernel:-intkernel,intkernel:-intkernel], (1, Tile_Size, Tile_Size, 1)).astype('uint8')
                
                # Send it to the Writerator
                writerator.send(reshape_img)

            # Close the image
            bw.close_image()

    # Always close the JavaBridge
    finally:
        logger.info('Closing the javabridge...')
        jutil.kill_vm()
           
