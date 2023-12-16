"""Helper functions for tests."""

import pathlib
from typing import Optional

import bfio
import numpy
import pytest
from polus.plugins.visualization.precompute_slide.utils import ImageType
from polus.plugins.visualization.precompute_slide.utils import PyramidType

FixtureReturnType = tuple[
    pathlib.Path,  # input dir
    pathlib.Path,  # output dir
    pathlib.Path,  # input image path
    ImageType,
    PyramidType,
]


def gen_image(
    data_dir: pathlib.Path,
    request: pytest.FixtureRequest,
    name: Optional[str] = None,
) -> FixtureReturnType:
    """Generate an image for combination of any user defined params.

    Returns:
        Path to the input directory.
        Path to the output directory.
        Path to the input image.
        Image type.
        Pyramid type.
    """
    image_y: int
    image_x: int
    image_type: ImageType
    pyramid_type: PyramidType
    (image_y, image_x, image_type, pyramid_type) = request

    input_dir = data_dir.joinpath("inp_dir")
    input_dir.mkdir(exist_ok=True)

    output_dir = data_dir.joinpath("out_dir")
    output_dir.mkdir(exist_ok=True)

    # generate image data
    image_name = "test_image" if name is None else name
    if image_type == ImageType.Segmentation:
        image_path = create_label_image(
            input_dir=input_dir,
            image_y=image_y,
            image_x=image_x,
            image_name=image_name,
        )
    else:  # image_type == ImageType.Intensity
        image_path = create_intensity_image(
            input_dir=input_dir,
            image_y=image_y,
            image_x=image_x,
            image_name=image_name,
        )

    return (input_dir, output_dir, image_path, image_type, pyramid_type)


def _create_centered_square(image_y: int, image_x: int) -> numpy.ndarray:
    """Create a simple image of a centered white square over a black background.

    Args:
        image_y: Size of the image in the y dimension.
        image_x: Size of the image in the x dimension.

    Returns:
        Image data.
    """
    data = numpy.zeros((image_y, image_x), dtype=numpy.uint8)
    fill_value = 255
    center_y = image_y // 2
    center_x = image_x // 2
    data[
        center_y - center_y // 2 : center_y + center_y // 2,
        center_x - center_x // 2 : center_x + center_x // 2,
    ] = fill_value
    return data


def _add_noise(labels: numpy.ndarray) -> numpy.ndarray:
    """Adds random poisson noise to an image based on the intensity of each pixel.

    Args:
        labels: Image data.

    Returns:
        Image data with noise.
    """
    mask = labels > 0
    intensity = numpy.random.poisson(  # noqa: NPY002
        labels.astype(numpy.float32),
    ).astype(numpy.float32)
    intensity = intensity / intensity.max(initial=0)
    intensity[~mask] = 0
    return intensity


def create_label_image(
    input_dir: pathlib.Path,
    image_x: int,
    image_y: int,
    image_name: str,
) -> pathlib.Path:
    """Create a simple image of a centered white square over a black background.

    Args:
        input_dir: Path to the input directory.
        image_x: Size of the image in the x dimension.
        image_y: Size of the image in the y dimension.
        image_name: Name of the image to create.

    Returns:
        Path to the created image.
    """
    image_data = _create_centered_square(image_y, image_x)
    image_path = input_dir.joinpath(f"{image_name}.ome.tif")

    with bfio.BioWriter(image_path) as writer:
        writer.X = image_x
        writer.Y = image_y
        writer.dtype = image_data.dtype
        writer[:] = image_data[:]

    return image_path


def create_intensity_image(
    input_dir: pathlib.Path,
    image_y: int,
    image_x: int,
    image_name: str,
) -> pathlib.Path:
    """Create a simple image of a centered white square over a black background.

    Args:
        input_dir: Path to the input directory.
        image_y: Size of the image in the y dimension.
        image_x: Size of the image in the x dimension.
        image_name: Name of the image to create.

    Returns:
        Path to the created image.
    """
    image_data = _add_noise(_create_centered_square(image_y, image_x))
    image_path = input_dir.joinpath(f"{image_name}.ome.tif")

    with bfio.BioWriter(image_path) as writer:
        writer.Y = image_y
        writer.X = image_x
        writer.dtype = image_data.dtype
        writer[:] = image_data[:]

    return image_path


def next_level_shape(
    shape: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    """Return the shape of the next level of a pyramid.

    Args:
        shape: Shape of the current level.

    Returns:
        Shape of the next level.
    """
    return tuple(map(next_axis_len, shape))  # type: ignore[return-value]


def next_axis_len(axis_len: int) -> int:
    """Return the length of the next axis in a pyramid.

    Args:
        axis_len: Length of the current axis.

    Returns:
        Length of the next axis.
    """
    if axis_len == 1:
        return 1
    return axis_len // 2 if axis_len % 2 == 0 else (axis_len + 1) // 2


def next_segmentation_image(image: numpy.ndarray) -> numpy.ndarray:
    """Return the next segmentation image in a pyramid.

    Args:
        image: Image to downsample.

    Returns:
        Downsampled image.
    """
    return faster_mode2(image[0, 0, 0, :, :])[
        numpy.newaxis,
        numpy.newaxis,
        numpy.newaxis,
        :,
        :,
    ]


def faster_mode2(image: numpy.ndarray) -> numpy.ndarray:
    """Return the mode of a 2D image.

    Args:
        image: Image to take the mode of.

    Returns:
        Mode of the image.
    """
    if image.ndim != 2:
        msg = f"Image must be 2D but got {image.shape}"
        raise ValueError(msg)
    # if the image has an odd number columns, then we need to pad it with a
    # column, repeating the last column

    if image.shape[1] % 2 == 1:
        image = numpy.concatenate(
            [
                image,
                image[:, -1:],
            ],
            axis=1,
        )

    # if the image has an odd number of rows, then we need to pad it with a row,
    # repeating the last row
    if image.shape[0] % 2 == 1:
        image = numpy.concatenate(
            [
                image,
                image[-1:, :],
            ],
            axis=0,
        )

    top_left = image[0:-1:2, 0:-1:2]
    top_right = image[0:-1:2, 1::2]
    bottom_left = image[1::2, 0:-1:2]
    bottom_right = image[1::2, 1::2]

    # stack the 4 pixels in each 2x2 square into a 3D array
    squares = numpy.stack(
        [top_left, top_right, bottom_left, bottom_right],
        axis=2,
    )

    # reshape the 3D array into a 2D array where each row is the four pixels in
    # a 2x2 square
    squares = squares.reshape(-1, 4)

    # sort each row by non-increasing intensity
    sorted_squares = numpy.sort(squares, axis=1)[:, ::-1]

    # also get the indices in the original array of the sorted pixels
    sorted_indices = numpy.argsort(squares, axis=1)[:, ::-1]

    # for every element in a row, find the first element that is equal to the
    # element to its right.
    # if there is no such element, then the mode is the first element in the
    # row.
    equal_to_right = (sorted_squares[:, :-1] == sorted_squares[:, 1:]).astype(
        numpy.uint8,
    )

    # get the indices of the first element in each row that is equal to the
    # element to its right
    equal_to_right_indices = numpy.argmax(equal_to_right, axis=1)

    # remap indices to the original array
    equal_to_right_indices = sorted_indices[
        numpy.arange(sorted_indices.shape[0]),
        equal_to_right_indices,
    ]

    # Get the element from each row at that index
    mode = sorted_squares[numpy.arange(sorted_squares.shape[0]), equal_to_right_indices]

    # reshape the mode array into the half the shape of the original image after
    # padding
    return mode.reshape(image.shape[0] // 2, image.shape[1] // 2)


def next_intensity_image(image: numpy.ndarray) -> numpy.ndarray:
    """Return the next intensity image in a pyramid.

    Args:
        image: Image to downsample.

    Returns:
        Downsampled image.
    """
    # Take the mean of each 2x2 square and make it the new pixel value in the
    # downsampled image.

    next_shape = next_level_shape(image.shape)
    next_image = numpy.zeros(
        next_shape,
        dtype=image.dtype,
    )
    for i in range(0, image.shape[0], 2):
        for j in range(0, image.shape[1], 2):
            next_image[0, 0, 0, i // 2, j // 2] = numpy.mean(
                image[0, 0, 0, i : i + 2, j : j + 2],
            )
    return next_image
