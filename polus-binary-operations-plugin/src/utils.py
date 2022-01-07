import logging, traceback
import os

from bfio import BioReader, BioWriter
import numpy as np

import matplotlib.pyplot as plt

import cv2

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
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

def iterate_tiles(shape, window_size, step_size):

    for y1 in range(0, shape[0], step_size):
        for x1 in range(0, shape[1], step_size):

            y2_window = min(shape[0], y1+window_size)
            x2_window = min(shape[1], x1+window_size)

            y2_step   = min(shape[0], y1+step_size)
            x2_step   = min(shape[1], x1+step_size)

            window_slice = (slice(y1, y2_window), slice(x1, x2_window), 
                                slice(0,1), slice(0,1), slice(0,1))
            step_slice   = (slice(y1, y2_step), slice(x1, x2_step), 
                                slice(0,1), slice(0,1), slice(0,1))

            yield window_slice, step_slice


def binary_operation(input_path: str,
                     output_path: str,
                     function,
                     extra_arguments,
                     override: bool,
                     extra_padding: int =512,
                     kernel: int =None):
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


    try: 
        # Read the image
        logger.info(f"\n\n OPERATING ON {os.path.basename(input_path)}")
        logger.debug(f"Input Path: {input_path}")
        logger.debug(f"Output Path: {output_path}")

        with BioReader(input_path, backend='java') as br:
            with BioWriter(output_path, backend='java',
                             metadata=br.metadata, dtype=br.dtype) as bw:
                
                bw_numpy = np.zeros(bw.shape).squeeze()
                assert br.shape == bw.shape
                bfio_shape = br.shape
                logger.info(f"Shape of BioReader&BioWriter (YXZCT): {bfio_shape}")
                logger.info(f"DataType of BioReader&BioWriter: {br.dtype}")

                step_size = br._TILE_SIZE
                window_size = step_size + (2*extra_padding)

                for window_slice, step_slice in iterate_tiles(shape       = bfio_shape, 
                                                              window_size = window_size,
                                                              step_size   = step_size):
                    
                    logger.debug("\n SLICES...")
                    logger.debug(f"Window Y: {window_slice[0]}")
                    logger.debug(f"Window X: {window_slice[1]}")
                    logger.debug(f"Step Y: {step_slice[0]}")
                    logger.debug(f"Step X: {step_slice[1]}")

                    tile_readarray = br[window_slice]
                    tile_labels = np.unique(tile_readarray)

                    tile_writearray = np.zeros(tile_readarray.shape).astype(br.dtype)
                    for label in tile_labels[1:]: # do not want to include zero (the background)
                        tile_binaryarray = (tile_readarray == label).astype(np.uint16)
                        if callable(function):
                            tile_binaryarray_modified = function(tile_binaryarray, 
                                                                 kernel=kernel, 
                                                                 n=extra_arguments)
                            tile_binaryarray_modified[tile_binaryarray_modified == 1] = label
                        
                        if override == True:
                            idx = (tile_binaryarray != tile_binaryarray_modified) | (tile_readarray == label)
                        else:
                            idx = ((tile_readarray == label) | (tile_readarray == 0)) & (tile_writearray == 0)

                        tile_writearray[idx] = tile_binaryarray_modified[idx].astype(br.dtype)
                        
                    logger.debug(f"Input Tile has {len(np.unique(tile_writearray)[1:])} labels & " + \
                                 f"Output Tile has {len(tile_labels[1:])} labels")
                    
                    # bw[step_slice] = tile_writearray[0:step_size, 0:step_size]
                    bw_numpy[step_slice[:2]] = tile_writearray[0:step_size, 0:step_size]

                plt.imshow(bw_numpy)
                plt.savefig(output_path + ".jpg")

        return output_path

    except Exception as e:
        raise ValueError(f"Something went wrong: {traceback.print_exc(e)}")
