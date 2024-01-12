"""Binary operations and processing utilities."""

import logging
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("utils")

TileTuple = tuple[slice, slice, slice, slice, slice]


def invert(image: np.ndarray, **_: Any) -> np.ndarray:  # noqa: ANN401
    """Invert an image."""
    return (~(image > 0.5)).astype(np.uint8)  # noqa: PLR2004


def dilate(image: np.ndarray, kernel: Any, n: int = 1) -> np.ndarray:  # noqa: ANN401
    """Perform a binary dilation.

    This function uses opencv's dilation function.

    Args:
        image: Image to dilate.
        kernel: The opencv structuring element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The dilated image
    """
    return cv2.dilate(image, kernel, iterations=n)


def erode(image: np.ndarray, kernel: Any, n: int = 1) -> np.ndarray:  # noqa: ANN401
    """Performa  binary erosion.

    This function uses opencv's erosion function.

    Args:
        image: Image to erode.
        kernel: The opencv structuring element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The eroded image.
    """
    return cv2.erode(image, kernel, iterations=n)


def open_(image: np.ndarray, kernel: int, _: int = 1) -> np.ndarray:
    """Perform a binary opening operation.

    The opening operation is similar to running an erosion followed by a dilation.

    Args:
        image: Image to perform binary opening on.
        kernel: The opencv strucuturing element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The opened image.
    """
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def close_(image: np.ndarray, kernel: int, _: int = 1) -> np.ndarray:
    """Perform a binary closing operation.

    The opening operation is similar to running a dilation followed by an erosion.

    Args:
        image: Image to perform binary closing on.
        kernel: The opencv strucuturing element.
        n: Number of iterations. Defaults to 1.

    Returns:
        The closed image.
    """
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def morphgradient(
    image: np.ndarray,
    kernel: Any,  # noqa: ANN401
    _: int = 1,
) -> np.ndarray:
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
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def fill_holes(image: np.ndarray, *_: Any) -> np.ndarray:  # noqa: ANN401
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
        image,
        mode=cv2.RETR_CCOMP,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    for cnt in contour:
        cv2.drawContours(image, [cnt], 0, 1, -1)

    return image.astype(image_dtype)


def skeletonize(
    image: np.ndarray,
    kernel: Any,  # noqa: ANN401
    _: int = 0,
) -> np.ndarray:
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


def tophat(image: np.ndarray, kernel: Any, _: int = 0) -> np.ndarray:  # noqa: ANN401
    """Difference between the input image and opening of the image.

    Args:
        image: Image to perform tophat on.
        kernel: The opencv strucuturing element.
        n: Not used.

    Returns:
        An image with tophat operation performed on it.
    """
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def blackhat(
    image: np.ndarray,
    kernel: Any = None,  # noqa: ANN401
    _: int = 0,
) -> np.ndarray:
    """Difference between the closing of the input image and input image.

    Args:
        image: Image to perform blackhat on.
        kernel: The opencv strucuturing element.
        n: Not used.

    Returns:
        An image with blackhat performed on it.
    """
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def remove_small(
    image: np.ndarray,
    _: Any = None,  # noqa: ANN401
    n: int = 2,
) -> np.ndarray:
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

    return uniques[inverse].reshape(image.shape)


def remove_large(
    image: np.ndarray,
    _: Any = None,  # noqa: ANN401
    n: int = 0,
) -> np.ndarray:
    """Remove small objects from the image.

    Removes all objects in the image that have an area larger than the threshold.

    Args:
        image: Image to remove small objects from.
        kernel: Not used.
        n: Threshold size under which objects will be removed. Defaults to 2.

    Returns:
        An image with small objects removed.
    """
    if n <= 0:
        msg = "n must be a positive, non-zero value"
        logger.error(msg)
        raise ValueError(msg)
    uniques, inverse, counts = np.unique(image, return_inverse=True, return_counts=True)

    uniques[counts > n] = 0

    return uniques[inverse].reshape(image.shape)


def iterate_tiles(
    shape: tuple,
    window_size: int,
    step_size: int,
) -> Generator[tuple[TileTuple, TileTuple], None, None]:
    """Iterate through tiles of an image.

    Args:
        shape: Shape of the input
        window_size: Width and Height of the Tile
        step_size: The size of steps that are iterated though the tiles

    Returns:
        Two tuples of slices that represent the window size and step size in 5 dimenions
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
