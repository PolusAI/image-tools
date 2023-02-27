"""Binary operations and processing utilities."""

import logging
from typing import Any, Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger("utils")

TileTuple = Tuple[slice, slice, slice, slice, slice]


def invert(image: np.ndarray, **kwargs) -> np.ndarray:
    """Invert an image."""
    return (~(image > 0.5)).astype(np.uint8)


def dilate(image: np.ndarray, kernel: Any, n: int = 1) -> np.ndarray:
    """Perform a binary dilation.

    This function uses opencv's dilation function.

    Args:
        image: Image to dilate.
        kernel: The opencv structuring element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The dilated image
    """
    dilatedimg = cv2.dilate(image, kernel, iterations=n)
    return dilatedimg


def erode(image: np.ndarray, kernel: Any, n: int = 1) -> np.ndarray:
    """Performa  binary erosion.

    This function uses opencv's erosion function.

    Args:
        image: Image to erode.
        kernel: The opencv structuring element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The eroded image.
    """
    erodedimg = cv2.erode(image, kernel, iterations=n)
    return erodedimg


def open_(image: np.ndarray, kernel: int, n: int = 1) -> np.ndarray:
    """Perform a binary opening operation.

    The opening operation is similar to running an erosion followed by a dilation.

    Args:
        image: Image to perform binary opening on.
        kernel: The opencv strucuturing element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The opened image.
    """
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openimg


def close_(image: np.ndarray, kernel: int, n: Any = 1) -> np.ndarray:
    """Perform a binary closing operation.

    The opening operation is similar to running a dilation followed by an erosion.

    Args:
        image: Image to perform binary closing on.
        kernel: The opencv strucuturing element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The closed image.
    """
    closeimg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closeimg


def morphgradient(image: np.ndarray, kernel: Any, n: int = 1) -> np.ndarray:
    """Calculate the morphological gradient.

    The morphological gradient is the difference between the dilated and eroded images.
    It effectively creates object outlines.

    Args:
        image: Image to perform morphological gradient on.
        kernel: The opencv strucuturing element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The morphological gradient of the input image.
    """
    mg = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return mg


def fill_holes(image: np.ndarray, kernel: Any = None, n: int = 0) -> np.ndarray:
    """Fill holes in objects.

    This algorithm was modeled off this Stack Overflow answer.
    https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object

    Args:
        image: Image to perform morphological gradient on.
        kernel: Not used.
        n: Not used.

    Returns:
        An image with holes inside of objects filled in.
    """
    image_dtype = image.dtype
    image = cv2.convertScaleAbs(image)
    contour, _ = cv2.findContours(
        image, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contour:
        cv2.drawContours(image, [cnt], 0, 1, -1)

    image = image.astype(image_dtype)

    return image


def skeletonize(image: np.ndarray, kernel: Any, n: int = 0) -> np.ndarray:
    """Skeletonize objects in an image.

    This algorithm was inspired by the algorithm described here:
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm

    Args:
        image: Image to perform morphological gradient on.
        kernel: The opencv strucuturing element.
        n: Not used.

    Returns:
        An image with all objects skeletonized.
    """
    done = False
    size = np.size(image)
    skel = np.zeros(image.shape, image.dtype)

    while not done:
        erode = cv2.erode(image, kernel)
        temp = cv2.dilate(erode, kernel)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = erode.copy()
        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True

    return skel


def tophat(image: np.ndarray, kernel: Any, n: int = 0) -> np.ndarray:
    """Difference between the input image and opening of the image.

    Args:
        image: Image to perform tophat on.
        kernel: The opencv strucuturing element.
        n: Not used.

    Returns:
        An image with tophat operation performed on it.
    """
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat


def blackhat(image: np.ndarray, kernel: Any = None, n: int = 0) -> np.ndarray:
    """Difference between the closing of the input image and input image.

    Args:
        image: Image to perform blackhat on.
        kernel: The opencv strucuturing element.
        n: Not used.

    Returns:
        An image with blackhat performed on it.
    """
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return blackhat


def remove_small(image: np.ndarray, kernel: Any = None, n: int = 2) -> np.ndarray:
    """Remove small objects from the image.

    Removes all objects in the image that have an area larger than the threshold.

    Args:
        image: Image to remove small objects from.
        kernel: Not used.
        n: Threshold size under which objects will be removed. Defaults to 2.

    Returns:
        An image with small objects removed.
    """
    uniques, inverse, counts = np.unique(image, return_inverse=True, return_counts=True)

    uniques[counts < n] = 0

    image_out = uniques[inverse].reshape(image.shape)

    return image_out


def remove_large(image: np.ndarray, kernel: Any = None, n: int = 0) -> np.ndarray:
    """Remove small objects from the image.

    Removes all objects in the image that have an area larger than the threshold.

    Args:
        image: Image to remove small objects from.
        kernel: Not used.
        n: Threshold size under which objects will be removed. Defaults to 2.

    Returns:
        An image with small objects removed.
    """
    assert n > 0, "n must be a positive, non-zero value"
    uniques, inverse, counts = np.unique(image, return_inverse=True, return_counts=True)

    uniques[counts > n] = 0

    image_out = uniques[inverse].reshape(image.shape)

    return image_out


def iterate_tiles(
    shape: tuple, window_size: int, step_size: int
) -> Generator[Tuple[TileTuple, TileTuple], None, None]:
    """Iterate through tiles of an image.

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
            y2_window = min(shape[0], y1 + window_size)
            x2_window = min(shape[1], x1 + window_size)

            y2_step = min(shape[0], y1 + step_size)
            x2_step = min(shape[1], x1 + step_size)

            window_slice = (
                slice(y1, y2_window),
                slice(x1, x2_window),
                slice(0, 1),
                slice(0, 1),
                slice(0, 1),
            )
            step_slice = (
                slice(y1, y2_step),
                slice(x1, x2_step),
                slice(0, 1),
                slice(0, 1),
                slice(0, 1),
            )

            logger.debug("\n SLICES...")
            logger.debug(f"Window Y: {window_slice[0]}")
            logger.debug(f"Window X: {window_slice[1]}")
            logger.debug(f"Step Y: {step_slice[0]}")
            logger.debug(f"Step X: {step_slice[1]}")

            yield window_slice, step_slice


# def binary_operation(
#     input_path: str,
#     output_path: str,
#     function,
#     extra_arguments: Any,
#     override: bool,
#     operation: str,
#     extra_padding: int = 512,
#     kernel: int = None,
# ) -> str:
#     """
#     This function goes through the images and calls the appropriate binary operation

#     Parameters
#     ----------
#     input_path : str
#         Location of image
#     output_path : str
#         Location for BioWriter
#     function: str
#         The binary operation to dispatch on image
#     operation: str
#         The name of the binary operation to dispatch on image
#     extra_arguments : int
#         Extra argument(s) for the binary operation that is called
#     override: bool
#         Specifies whether previously saved instance labels in the
#         output can be overriden.
#     extra_padding : int
#         The extra padding around each tile so that
#         binary operations do not skewed around the edges.
#     kernel : cv2 object
#         The kernel used for most binary operations

#     Returns
#     -------
#     output_path : str
#         Location of BioWriter for logger in main.py

#     """

#     try:
#         # Read the image and log its information
#         logger.info(f"\n\n OPERATING ON {os.path.basename(input_path)}")
#         logger.debug(f"Input Path: {input_path}")
#         logger.debug(f"Output Path: {output_path}")

#         with BioReader(input_path) as br:
#             with BioWriter(output_path, metadata=br.metadata) as bw:
#                 assert br.shape == bw.shape
#                 bfio_shape: tuple = br.shape
#                 logger.info(f"Shape of BioReader&BioWriter (YXZCT): {bfio_shape}")
#                 logger.info(f"DataType of BioReader&BioWriter: {br.dtype}")

#                 step_size: int = br._TILE_SIZE
#                 window_size: int = step_size + (2 * extra_padding)

#                 for window_slice, step_slice in iterate_tiles(
#                     shape=bfio_shape, window_size=window_size, step_size=step_size
#                 ):
#                     # info on the Slices for debugging
#                     logger.debug("\n SLICES...")
#                     logger.debug(f"Window Y: {window_slice[0]}")
#                     logger.debug(f"Window X: {window_slice[1]}")
#                     logger.debug(f"Step Y: {step_slice[0]}")
#                     logger.debug(f"Step X: {step_slice[1]}")

#                     # read a tile of BioReader
#                     tile_readarray = br[window_slice]

#                     # get unique labels in the tile
#                     tile_readlabels = np.unique(tile_readarray)
#                     assert np.all(
#                         tile_readlabels >= 0
#                     ), f"There are negative numbers in the input tile: {tile_readlabels[tile_readlabels < 0]}"
#                     tile_readlabels = tile_readlabels[
#                         tile_readlabels > 0
#                     ]  # not just [1:], because there might not be a background

#                     if len(tile_readlabels) > 1:
#                         assert (
#                             operation != "inversion"
#                         ), "Image has multiple labels, you cannot use inversion on this type of image!"

#                     # BioWriter handles tiles of 1024, need to be able to manipulate output,
#                     # therefore initialize output numpy array
#                     tile_writearray = np.zeros(tile_readarray.shape).astype(br.dtype)

#                     # iterate through labels in tile
#                     for (
#                         label
#                     ) in (
#                         tile_readlabels
#                     ):  # do not want to include zero (the background)
#                         tile_binaryarray = (tile_readarray == label).astype(
#                             np.uint16
#                         )  # get another image with just the one label

#                         # if the operation is callable
#                         if callable(function):
#                             tile_binaryarray_modified = function(
#                                 tile_binaryarray, kernel=kernel, n=extra_arguments
#                             )  # outputs an image with just 0 and 1
#                             tile_binaryarray_modified[
#                                 tile_binaryarray_modified == 1
#                             ] = label  # convert the 1 back to label value

#                         if override == True:
#                             # if completely overlapping another instance segmentation in output is okay
#                             # take the difference between the binary and modified binary (in dilation it would be the borders that were added on)
#                             # and the input label
#                             idx = (tile_binaryarray != tile_binaryarray_modified) | (
#                                 tile_readarray == label
#                             )
#                         else:
#                             # otherwise if its not
#                             # take the input label and the common background between the input and output arrays
#                             idx = (
#                                 (tile_readarray == label) | (tile_readarray == 0)
#                             ) & (tile_writearray == 0)

#                         # save the one label to the output
#                         tile_writearray[idx] = tile_binaryarray_modified[idx].astype(
#                             br.dtype
#                         )

#                     tile_writelabels = np.unique(tile_writearray)
#                     tile_writelabels = tile_writelabels[tile_writelabels > 0]
#                     # if override is set to False, make sure that these values equal each other
#                     logger.debug(
#                         f"Input Tile has {len(tile_writelabels)} labels & "
#                         + f"Output Tile has {len(tile_readlabels)} labels"
#                     )
#                     logger.debug(f"INPUT TILE VALUES:  {tile_writelabels}")
#                     logger.debug(f"OUTPUT TILE VALUES: {tile_readlabels}")

#                     # finalize the output
#                     bw[step_slice] = tile_writearray[0:step_size, 0:step_size]

#         return output_path

#     except Exception as e:
#         raise ValueError(f"Something went wrong: {e}")
