import logging
import os

from typing import Any
from typing import Tuple

from bfio import BioReader, BioWriter
import numpy as np

import cv2

POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
logger.setLevel(POLUS_LOG)


def invert_binary(image  : np.ndarray, 
                  kernel : int = None, 
                  n      : Any = None) -> np.ndarray:
    """
    This function inverts the binary image. 
    The 0s get mapped to 1.
    The 1s get mapped to 0.
    """
    invertedimg = np.zeros(image.shape).astype('uint8')
    invertedimg = 1 - image
    return invertedimg

def dilate_binary(image  : np.ndarray, 
                  kernel : int = None, 
                  n      : int = None) -> np.ndarray:
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

def erode_binary(image  : np.ndarray, 
                kernel : int = None, 
                n      : int = None) -> np.ndarray:
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

def open_binary(image  : np.ndarray, 
                kernel : int = None, 
                n      : Any = None) -> np.ndarray:
    """ 
    An opening operation is similar to applying an erosion 
    followed by a dilation.  It removes small objects/noise in the 
    background of the images.
    """
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openimg

def close_binary(image  : np.ndarray, 
                 kernel : int = None, 
                 n      : Any = None) -> np.ndarray:
    """
    A closing operation is similar to applying a dilation
    followed by an erosion. It is useful in closing small holes
    inside the foreground objects, or small black points inside
    the image. 
    """
    closeimg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closeimg

def morphgradient_binary(image  : np.ndarray, 
                         kernel : int = None, 
                         n      : Any = None) -> np.ndarray:
    """
    This operation is the difference between dilation and 
    erosion of an image.  It creates an outline of the 
    foreground object.
    """
    mg = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return mg

def fill_holes_binary(image  : np.ndarray, 
                      kernel : int = None, 
                      n      : Any = None) -> np.ndarray:
    """
    This function fills segments.  It finds countours in 
    the image, and then fills it with black, and inverts
    it back
    https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
    """

    image_dtype = image.dtype
    image = cv2.convertScaleAbs(image)
    contour,hier = cv2.findContours(image,mode=cv2.RETR_CCOMP,method=cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(image,[cnt],0,1,-1)

    image = image.astype(image_dtype)

    return image

def skeleton_binary(image  : np.ndarray, 
                    kernel : int = None, 
                    n      : Any = None) -> np.ndarray:
    """ 
    This operation reduces the  foreground regions in a binary image 
    to a skeletal remnant that largely preserves the extent and 
    connectivity of the original region while throwing away most of 
    the original foreground pixels.
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm
    """
    done = False
    size = np.size(image)
    skel = np.zeros(image.shape,image.dtype)

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

def tophat_binary(image  : np.ndarray, 
                  kernel : int = None, 
                  n      : Any = None) -> np.ndarray:
    """
    It is the difference between input image and
    opening of the image
    """
    tophat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
    return tophat

def blackhat_binary(image  : np.ndarray, 
                    kernel : int = None, 
                    n      : Any = None) -> np.ndarray:
    """
    It is the difference between the closing of 
    the input image and input image.
    """
    blackhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def areafiltering_remove_smaller_objects_binary(image : np.ndarray, 
                                                kernel : int = None, 
                                                n      : int = None) -> np.ndarray:
    """ 
    Removes all objects in the image that have an area larger than 
    the threshold specified.
    
    Additional Arguments
    --------------------
    n : int
        Specifies the threshold.
    """

    image_dtype = image.dtype
    image = cv2.convertScaleAbs(image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    af = np.zeros((image.shape))
    count_removed = 0
    logger.debug("{} ROI in tile".format(nb_components))
    for i in range(0, nb_components):
        if sizes[i] >= n:
            af[output == i+1] = 1
            count_removed = count_removed + 1
    logger.debug("{} ROI removed in tile".format(count_removed))

    af = af.astype(image_dtype)
    return af

def areafiltering_remove_larger_objects_binary(image  : np.ndarray, 
                                               kernel : int = None, 
                                               n      : int = None) -> np.ndarray:
    """ 
    Removes all objects in the image that have an area smaller than 
    the threshold specified.
    
    Arguments
    ---------
    n : int
        Specifies the threshold.
    """

    image_dtype = image.dtype
    image = cv2.convertScaleAbs(image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    af = np.zeros((image.shape))
    count_removed = 0
    logger.debug("{} ROI in tile".format(nb_components))
    for i in range(0, nb_components):
        if sizes[i] <= n:
            af[output == i+1] = 1
            count_removed = count_removed + 1
    logger.debug("{} ROI removed in tile".format(count_removed))

    af = af.astype(image_dtype)
    return af

def iterate_tiles(shape       : tuple, 
                  window_size : int, 
                  step_size   : int) -> tuple:

    """ This function helps to iterate through tiles of the input 
        and output image by yielding the corresponding slices. 
        
    Arguments
    ---------
    shape : tuple
        Shape of the input
    window_size : int
        Width and Height of the Tile
    step_size : int 
        The size of steps that are iterated though the tiles

    Returns
    -------
    window_slice : slice
        5 dimensional slice that specifies the indexes of the window tile
    step_slice : slice
        5 dimension slice that specifies the indexes of the step tile
    """

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
                     extra_arguments: Any,
                     override: bool,
                     operation: str, 
                     extra_padding: int = 512,
                     kernel: int = None) -> str:
    """
    This function goes through the images and calls the appropriate binary operation

    Parameters
    ----------
    input_path : str
        Location of image
    output_path : str
        Location for BioWriter
    function: str
        The binary operation to dispatch on image
    operation: str
        The name of the binary operation to dispatch on image
    extra_arguments : int
        Extra argument(s) for the binary operation that is called
    override: bool
        Specifies whether previously saved instance labels in the 
        output can be overriden.
    extra_padding : int
        The extra padding around each tile so that
        binary operations do not skewed around the edges. 
    kernel : cv2 object
        The kernel used for most binary operations

    Returns
    -------
    output_path : str
        Location of BioWriter for logger in main.py

    """


    try: 
        # Read the image and log its information
        logger.info(f"\n\n OPERATING ON {os.path.basename(input_path)}")
        logger.debug(f"Input Path: {input_path}")
        logger.debug(f"Output Path: {output_path}")

        with BioReader(input_path, backend='java') as br:
            with BioWriter(output_path, backend='java',
                             metadata=br.metadata, dtype=br.dtype) as bw:
                
                assert br.shape == bw.shape
                bfio_shape: tuple = br.shape
                logger.info(f"Shape of BioReader&BioWriter (YXZCT): {bfio_shape}")
                logger.info(f"DataType of BioReader&BioWriter: {br.dtype}")

                step_size: int = br._TILE_SIZE
                window_size: int = step_size + (2*extra_padding)

                for window_slice, step_slice in iterate_tiles(shape       = bfio_shape, 
                                                              window_size = window_size,
                                                              step_size   = step_size):
                    
                    # info on the Slices for debugging
                    logger.debug("\n SLICES...")
                    logger.debug(f"Window Y: {window_slice[0]}")
                    logger.debug(f"Window X: {window_slice[1]}")
                    logger.debug(f"Step Y: {step_slice[0]}")
                    logger.debug(f"Step X: {step_slice[1]}")

                    # read a tile of BioReader
                    tile_readarray = br[window_slice]

                    # get unique labels in the tile
                    tile_labels = np.unique(tile_readarray)
                    if len(tile_labels) > 2:
                        assert operation != "inversion", "Image has multiple labels, you cannot use inversion on this type of image!"

                    # BioWriter handles tiles of 1024, need to be able to manipulate output, 
                        # therefore initialize output numpy array
                    tile_writearray = np.zeros(tile_readarray.shape).astype(br.dtype)

                    # iterate through labels in tile
                    for label in tile_labels[1:]: # do not want to include zero (the background)

                        tile_binaryarray = (tile_readarray == label).astype(np.uint16) # get another image with just the one label 

                        # if the operation is callable
                        if callable(function):
                            tile_binaryarray_modified = function(tile_binaryarray, 
                                                                 kernel=kernel, 
                                                                 n=extra_arguments) # outputs an image with just 0 and 1
                            tile_binaryarray_modified[tile_binaryarray_modified == 1] = label # convert the 1 back to label value
                        
                        if override == True:
                            # if completely overlapping another instance segmentation in output is okay
                                # take the difference between the binary and modified binary (in dilation it would be the borders that were added on)
                                    # and the input label
                            idx = (tile_binaryarray != tile_binaryarray_modified) | (tile_readarray == label)
                        else:
                            # otherwise if its not
                                # take the input label and the common background between the input and output arrays
                            idx = ((tile_readarray == label) | (tile_readarray == 0)) & (tile_writearray == 0)

                        # save the one label to the output
                        tile_writearray[idx] = tile_binaryarray_modified[idx].astype(br.dtype)
                        
                    # if override is set to False, make sure that these values equal each other
                    logger.debug(f"Input Tile has {len(np.unique(tile_writearray)[1:])} labels & " + \
                                 f"Output Tile has {len(tile_labels[1:])} labels")
                    
                    
                    # finalize the output
                    bw[step_slice] = tile_writearray[0:step_size, 0:step_size]

        return output_path

    except Exception as e:
        raise ValueError(f"Something went wrong: {e}")
