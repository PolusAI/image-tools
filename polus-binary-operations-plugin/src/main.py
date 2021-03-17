from bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, traceback, multiprocessing

import numpy as np

from pathlib import Path
import os

import cv2

Tile_Size = 256

def invert_binary(image, kernel=None, n=None):
    """
    This function inverts the binary image. 
    The 0s get mapped to 1.
    The 1s get mapped to 0.
    """
    invertedimg = np.zeros(image.shape).astype('uint8')
    invertedimg = 1 - image
    return invertedimg

def dilate_binary(image, kernel=None, n=None): 
    """
    Increases the white region in the image, or the 
    foreground object increases.
    
    Additional Arguments:
    ---------------------
    n : int
        (iterations) The number of times to apply the dilation.
    """
    dilatedimg = cv2.dilate(image, kernel, iterations=n)
    return dilatedimg

def erode_binary(image, kernel=None, n=None):
    """
    Decreases the white region in the image, or the 
    foreground object decreases.
    
    Additional Arguments:
    ---------------------
    n : int
        (iterations) The number of times to apply the erosion.
    """
    erodedimg = cv2.erode(image, kernel, iterations=n)
    return erodedimg

def open_binary(image, kernel=None, n=None):
    """ 
    An opening operation is similar to applying an erosion 
    followed by a dilation.  It removes small objects/noise in the 
    background of the images.
    """
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openimg

def close_binary(image, kernel=None, n=None):
    """
    A closing operation is similar to applying a dilation
    followed by an erosion. It is useful in closing small holes
    inside the foreground objects, or small black points inside
    the image. 
    """
    closeimg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closeimg

def morphgradient_binary(image, kernel=None, n=None):
    """
    This operation is the difference between dilation and 
    erosion of an image.  It creates an outline of the 
    foreground object.
    """
    mg = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return mg

def skeleton_binary(image, kernel=None, n=None):
    """ 
    This operation reduces the  foreground regions in a binary image 
    to a skeletal remnant that largely preserves the extent and 
    connectivity of the original region while throwing away most of 
    the original foreground pixels.
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm
    """
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


def tophat_binary(image, kernel=None, n=None):
    """
    It is the difference between input image and
    opening of the image
    """
    tophat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
    return tophat

def blackhat_binary(image, kernel=None, n=None):
    """
    It is the difference between the closing of 
    the input image and input image.
    """
    blackhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def areafiltering_remove_smaller_objects_binary(image, kernel=None, n=None):
    """ 
    Removes all objects in the image that have an area larger than 
    the threshold specified.
    
    Additional Arguments
    --------------------
    n : int
        Specifies the threshold.
    """

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    af = np.ones((image.shape))
    count_removed = 0
    logger.info("{} ROI in tile".format(nb_components))
    for i in range(0, nb_components):
        if sizes[i] >= n:
            af[output == i+1] = 1
            count_removed = count_removed + 1
    logger.info("{} ROI removed in tile".format(count_removed))
    return af

def areafiltering_remove_larger_objects_binary(image, kernel=None, n=None):
    """ 
    Removes all objects in the image that have an area smaller than 
    the threshold specified.
    
    Additional Arguments
    --------------------
    n : int
        Specifies the threshold.
    """

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    af = np.zeros((image.shape))
    count_removed = 0
    logger.info("{} ROI in tile".format(nb_components))
    for i in range(0, nb_components):
        if sizes[i] <= n:
            af[output == i+1] = 1
            count_removed = count_removed + 1
    logger.info("{} ROI removed in tile".format(count_removed))
    return af


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
    parser.add_argument('--structuringshape', dest='struct_shape', type=str,
                        help='Shape of the structuring element can either be Elliptical, Rectangular, or Cross', required=True)

    # Extra arguments based on operation
    parser.add_argument('--kernelsize', dest='all_kernel', type=int, # not used for the area filtering
                        help='Kernel size that should be used for all operations', required=False)
    parser.add_argument('--ThresholdAreaRemoveLarge', dest='threshold_area_rm_large', type=int,
                        help='Area threshold of objects in image', required=False)
    parser.add_argument('--ThresholdAreaRemoveSmall', dest='threshold_area_rm_small', type=int,
                        help='Area threshold of objects in image', required=False)
    parser.add_argument('--IterationsDilation', dest='num_iterations_dilation', type=int,
                        help='Number of Iterations to apply operation', required=False)
    parser.add_argument('--IterationsErosion', dest='num_iterations_erosion', type=int,
                        help='Number of Iterations to apply operation', required=False)
    try:
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

        threshold_area_rm_large = args.threshold_area_rm_large
        threshold_area_rm_small = args.threshold_area_rm_small
        iterations_dilation = args.num_iterations_dilation
        iterations_erosion = args.num_iterations_erosion

        if 'filter_area_remove_large_objects' in operations:
            if threshold_area_rm_large == None:
                raise ValueError('Need to specify the maximum area of the segments to keep')

        if 'filter_area_remove_small_objects' in operations:
            if threshold_area_rm_small == None:
                raise ValueError('Need to specify the minimum area of the segments to keep')

        if 'dilation' in operations:
            if iterations_dilation == None:
                raise ValueError("Need to specify the number of iterations to apply the operation")

        if 'erosion' in operations:
            if iterations_erosion == None:
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
            'top_hat': tophat_binary,
            'black_hat': blackhat_binary,
            'filter_area_remove_large_objects': areafiltering_remove_larger_objects_binary,
            'filter_area_remove_small_objects': areafiltering_remove_smaller_objects_binary
        }

        # Additional arguments for each function
        dict_n_args = {
            'invertion': None,
            'opening': None,
            'closing': None,
            'morphological_gradient': None,
            'dilation': iterations_dilation,
            'erosion': iterations_erosion,
            'skeleton': None,
            'top_hat': None,
            'black_hat': None,
            'filter_area_remove_large_objects' : threshold_area_rm_large,
            'filter_area_remove_small_objects' : threshold_area_rm_small
        }
        

        # Start the javabridge with proper java logging
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir()]
        logger.info("Files in input directory: {}".format(inpDir_files))

        
        # Loop through files in inpDir image collection and process

        for f in inpDir_files:

            # Load an image
            image = os.path.join(inpDir, f)

            # Read the image
            br = BioReader(image)

            # Get the dimensions of the Image
            br_x, br_y, br_z, br_c, br_t = br.num_x(), br.num_y(), br.num_z(), br.num_c(), br.num_t()
            br_shape = (br_x, br_y, br_z, br_c, br_t)
            datatype = br.pixel_type()
            max_datatype_val = np.iinfo(datatype).max

            logger.info("Original Datatype {}: ({})".format(datatype, max_datatype_val))
            logger.info("Shape of Input (XYZCT): {}".format(br_shape))

            # Initialize Kernel
            kernel = cv2.getStructuringElement(structshape,(intkernel,intkernel))

            # Initialize Output
            newfile = os.path.join(outDir, f)
            bw = BioWriter(file_path=newfile, metadata=br.read_metadata())

            # Initialize the Python Generators to go through each "tile" of the image
            if (threshold_area_rm_large != None) or (threshold_area_rm_small != None):
                tsize = (Tile_Size * 2)
            else:
                tsize = Tile_Size + (2*intkernel)
            logger.info("Tile Size {}x{}".format(tsize, tsize))
            readerator = br.iterate(tile_stride=[Tile_Size, Tile_Size],tile_size=[tsize, tsize], batch_size=1)
            writerator = bw.writerate(tile_size=[Tile_Size, Tile_Size], tile_stride=[Tile_Size, Tile_Size], batch_size=1)
            next(writerator)

            for images,indices in readerator:
                # Extra tiles do not need to be calculated.
                    # Indices should range from -intkernel < index value < Image_Dimension + intkernel
                if (threshold_area_rm_large != None) or (threshold_area_rm_small != None):
                    if indices[0][0][0] == br_x - (Tile_Size//2):
                        continue
                    if indices[1][0][0] == br_y - (Tile_Size//2):
                        continue
                else:
                    if indices[0][0][0] == br_x - intkernel:
                        continue
                    if indices[1][0][0] == br_y - intkernel:
                        continue

                logger.info(indices)

                # Images are (1, Tile_Size, Tile_Size, 1)
                # Need to convert to (Tile_Size, Tile_Size) to be able to do operation
                images = np.squeeze(images)

                images[images == max_datatype_val] = 1
                
                # Initialize which function we are dispatching
                trans_image = None
                function = dispatch[operations]
                if callable(function):
                    trans_image = function(images, kernel=kernel, n=dict_n_args[operations])
                    trans_image = trans_image.astype(datatype)
                    trans_image[trans_image==1] = max_datatype_val

                # The image needs to be converted back to (1, Tile_Size_Tile_Size, 1) to write it
                reshape_img = None
                # Send it to the Writerator
                if (threshold_area_rm_large != None) or (threshold_area_rm_small != None):
                    reshape_img = np.reshape(trans_image[Tile_Size//2:-Tile_Size//2,Tile_Size//2:-Tile_Size//2], (1, Tile_Size, Tile_Size, 1))
                else:
                    reshape_img = np.reshape(trans_image[intkernel:-intkernel,intkernel:-intkernel], (1, Tile_Size, Tile_Size, 1))
                
                writerator.send(reshape_img)

            # Close the image
            bw.close_image()

    except:
        traceback.print_exc()
    # Always close the JavaBridge
    finally:
        logger.info('Closing the javabridge...')
        jutil.kill_vm()
