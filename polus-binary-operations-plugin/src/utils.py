from bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil

import logging, traceback
from pathlib import Path
import os

import cv2

import numpy as np

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


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

def fill_holes_binary(image, kernel=None, n=None):
    """
    This function fills segments.  It finds countours in 
    the image, and then fills it with black, and inverts
    it back
    https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
    """
    contour,hier = cv2.findContours(image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(image,[cnt],0,1,-1)

    return image

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
    skel = np.zeros(image.shape,np.uint8)

    while (not done):
        erode = cv2.erode(image,kernel)
        temp = cv2.dilate(erode,kernel)
        temp = cv2.subtract(image,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = erode.copy()
        zeros = size - cv2.countNonZero(image)
        if zeros==size:
            done = True

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

    af = np.zeros((image.shape))
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

def binary_operation(image,
                     output,
                     function,
                     extra_arguments,
                     extra_padding, 
                     kernel,
                     Tile_Size):
    """
    This function goes through the images and calls the appropriate binary operation

    Parameters
    ----------
    image : str
        Location of image
    function_to_call : str
        The binary operation to dispatch on image
    extra_arguments : int
        Extra argument(s) for the binary operation that is called
    extra_padding : int
        The extra padding around each tile so that
        binary operations do not skewed around the edges. 
    kernel : cv2 object
        The kernel used for most binary operations
    output : str
        Location for BioWriter
    Tile_Size : int
        Tile Size for reading images 
    """

    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    try: 
        # Read the image
        br = BioReader(image)

        # Get the dimensions of the Image
        br_x, br_y, br_z, br_c, br_t = br.num_x(), br.num_y(), br.num_z(), br.num_c(), br.num_t()
        br_shape = (br_x, br_y, br_z, br_c, br_t)
        datatype = br.pixel_type()
        max_datatype_val = np.iinfo(datatype).max

        logger.info("Original Datatype {}: ({})".format(datatype, max_datatype_val))
        logger.info("Shape of Input (XYZCT): {}".format(br_shape))

        # Initialize Output
        bw = BioWriter(file_path=output, metadata=br.read_metadata())

        # Initialize the Python Generators to go through each "tile" of the image
        tsize = Tile_Size + (2*extra_padding)
        logger.info("Tile Size {}x{}".format(tsize, tsize))
        readerator = br.iterate(tile_stride=[Tile_Size, Tile_Size],tile_size=[tsize, tsize], batch_size=1)
        writerator = bw.writerate(tile_size=[Tile_Size, Tile_Size], tile_stride=[Tile_Size, Tile_Size], batch_size=1)
        next(writerator)

        for images,indices in readerator:
            # Extra tiles do not need to be calculated.

            # Indices should range from -intkernel < index value < Image_Dimension + intkernel
            if (indices[0][0][0] == br_x - extra_padding) or (indices[1][0][0] == br_y - extra_padding):
                continue
            logger.info(indices)

            # Images are (1, Tile_Size, Tile_Size, 1)
            # Need to convert to (Tile_Size, Tile_Size) to be able to do operation
            images = np.squeeze(images)
            images[images == max_datatype_val] = 1
            
            # Initialize which function we are dispatching
            if callable(function):
                trans_image = function(images, kernel=kernel, n=extra_arguments)
                trans_image = trans_image.astype(datatype)
                trans_image[trans_image==1] = max_datatype_val

            # The image needs to be converted back to (1, Tile_Size_Tile_Size, 1) to write it
            reshape_img = np.reshape(trans_image[extra_padding:-extra_padding,extra_padding:-extra_padding], (1, Tile_Size, Tile_Size, 1))

            # Send it to the Writerator
            writerator.send(reshape_img)

        # Close the image
        bw.close_image()

    except:
        traceback.print_exc()

    # Always close the JavaBridge
    finally:
        jutil.kill_vm()
    
    
