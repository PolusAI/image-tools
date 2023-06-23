"""Common functions for the vector-field conversion plugins."""

import itertools

import numpy


def vector_norm(vector: numpy.ndarray, axis: int = -1) -> numpy.ndarray:
    """Normalizes `vector` to unit magnitude along the given axis.

    Args:
        vector: The vector to normalize.
        axis: The axis along which to normalize. Defaults to -1.

    Returns:
        The normalized vector.
    """
    norm = numpy.sqrt((vector**2).sum(axis=axis))

    return vector / (
        numpy.expand_dims(norm, axis=axis) + numpy.finfo(numpy.float32).eps
    )


class BoxFilterND:
    """An N-Dimensional Box Filter.

    This is a base class for efficient computation of box filters using integral
    images (aka summed area tables). This is O(1) complexity relative to kernel
    size.

    This base class simply calculates the local sum of pixel values.
    """

    def __init__(self, ndims: int, w: int = 3) -> None:
        """Initializes the box filter.

        Args:
            ndims: The number of dimensions of the input matrix. It is assumed the
            first index is used for different channels or images, and not included
            in the calculations.
            w: The window size for the box filter. Defaults to 3.
        """
        self.index = []
        self.sign = []

        should_add = (ndims - 1) % 2

        # Calculate the box filter indices
        for d in itertools.product(range(2), repeat=ndims - 1):
            if (sum(d) % 2) == should_add:
                self.sign.append(1)
            else:
                self.sign.append(-1)

            index = [slice(None)]

            for i in d:
                if i:
                    index.append(slice(w, None))
                else:
                    index.append(slice(None, -w))

            self.index.append(tuple(index))

    def __call__(self, image: numpy.ndarray) -> numpy.ndarray:
        """Calculates the box filter of the given image."""
        # Create the integral image
        integral_image = image
        for d in range(1, integral_image.ndim):
            integral_image = integral_image.cumsum(axis=d)

        # Calculate the box filter
        return sum(
            sign * integral_image[index] for sign, index in zip(self.sign, self.index)
        )


__all__ = [
    "BoxFilterND",
    "vector_norm",
]
