"""Utility functions for precompute_slide plugin."""

import copy
import enum
import logging
import os
import pathlib

import bfio
import numpy

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger(__file__)
logger.setLevel(POLUS_LOG)


class PyramidType(str, enum.Enum):
    """Pyramid type enumeration."""

    DeepZoom = "DeepZoom"
    Neuroglancer = "Neuroglancer"
    Zarr = "Zarr"


class ImageType(str, enum.Enum):
    """Image type enumeration."""

    Intensity = "Intensity"
    Segmentation = "Segmentation"


# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {
    "m": 10**9,
    "cm": 10**7,
    "mm": 10**6,
    "µm": 10**3,
    "nm": 1,
    "Å": 10**-1,
    "UnitsLength.MICROMETER": 10**3,
}

# Chunk Scale
CHUNK_SIZE = 1024


def _mode2(image: numpy.ndarray) -> numpy.ndarray:
    """Return the mode of a 2D image.

    Args:
        image: Image to take the mode of.

    Returns:
        Mode of the image.
    """
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

    # split the image into 4 pieces, each piece contains pixels from one corner
    # of a 2x2 square
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


def _avg2(image: numpy.ndarray) -> numpy.ndarray:
    """Average pixels together with optical field 2x2 and stride 2.

    Args:
        image: numpy array with only two dimensions (m,n)

    Returns:
        avg_img: numpy array with only two dimensions (round(m/2),round(n/2))
    """
    # Since we are adding pixel values, we need to update the pixel type
    # This helps to avoid integer overflow
    if image.dtype == numpy.uint8:
        dtype = numpy.uint16
    elif image.dtype == numpy.uint16:
        dtype = numpy.uint32
    elif image.dtype == numpy.uint32:
        dtype = numpy.uint64
    elif image.dtype == numpy.int8:
        dtype = numpy.int16
    elif image.dtype == numpy.int16:
        dtype = numpy.int32
    elif image.dtype == numpy.int32:
        dtype = numpy.int64
    else:
        dtype = image.dtype

    odtype = image.dtype
    image = image.astype(dtype)

    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2

    # Calculate the mean
    avg_img = numpy.zeros(
        numpy.ceil([d / 2 for d in image.shape]).astype(int),
        dtype=dtype,
    )
    avg_img[0 : y_max // 2, 0 : x_max // 2] = (
        image[0 : y_max - 1 : 2, 0 : x_max - 1 : 2]
        + image[1:y_max:2, 0 : x_max - 1 : 2]
        + image[0 : y_max - 1 : 2, 1:x_max:2]
        + image[1:y_max:2, 1:x_max:2]
    ) // 4

    # Fill in the final row if the image height is odd-valued
    if y_max != image.shape[0]:
        avg_img[-1, : x_max // 2] = (
            image[-1, 0 : x_max - 1 : 2] + image[-1, 1:x_max:2]
        ) // 2
    # Fill in the final column if the image width is odd-valued
    if x_max != image.shape[1]:
        avg_img[: y_max // 2, -1] = (
            image[0 : y_max - 1 : 2, -1] + image[1:y_max:2, -1]
        ) // 2
    # Fill in the lower right pixel if both image width and height are odd
    if y_max != image.shape[0] and x_max != image.shape[1]:
        avg_img[-1, -1] = image[-1, -1]

    return avg_img.astype(odtype)


def bfio_metadata_to_slide_info(
    image_path: pathlib.Path,
    out_path: pathlib.Path,  # noqa: ARG001
    stack_height: int,
    image_type: str,
    min_scale: int = 0,
) -> dict:
    """Generate a Neuroglancer info file from Bioformats metadata.

    Neuroglancer requires an info file in the root of the pyramid directory.
    All information necessary for this info file is contained in Bioformats
    metadata, so this function takes the metadata and generates the info file.

    Inputs:
        bfio_reader - A BioReader object
        outPath - Path to directory where pyramid will be generated
    Outputs:
        info - A dictionary containing the information in the info file
    """
    with bfio.BioReader(image_path, max_workers=1) as bfio_reader:
        # Get metadata info from the bfio reader
        sizes = [bfio_reader.X, bfio_reader.Y, stack_height]

        phys_x = bfio_reader.ps_x
        if None in phys_x:
            phys_x = (1000, "nm")

        phys_y = bfio_reader.ps_y
        if None in phys_y:
            phys_y = (1000, "nm")

        phys_z = bfio_reader.ps_z
        if None in phys_z:
            phys_z = ((phys_x[0] + phys_y[0]) / 2, phys_x[1])

        resolution = [phys_x[0] * UNITS[str(phys_x[1])]]
        resolution.append(phys_y[0] * UNITS[str(phys_y[1])])
        resolution.append(
            phys_z[0] * UNITS[str(phys_z[1])],
        )  # Just used as a placeholder
        dtype = str(numpy.dtype(bfio_reader.dtype))

    num_scales = int(numpy.ceil(numpy.log2(max(sizes))))

    # create a scales template, use the full resolution8
    scales = {
        "chunk_sizes": [[CHUNK_SIZE, CHUNK_SIZE, 1]],
        "encoding": "raw",
        "key": str(num_scales),
        "resolution": resolution,
        "size": sizes,
        "voxel_offset": [0, 0, 0],
    }

    # initialize the json dictionary
    info = {
        "data_type": dtype,
        "num_channels": 1,
        "scales": [scales],
        "type": image_type,
    }

    if image_type == "segmentation":
        info["segment_properties"] = "infodir"

    for i in reversed(range(min_scale, num_scales)):
        previous_scale = info["scales"][-1]  # type: ignore[index]
        current_scale = copy.deepcopy(previous_scale)
        current_scale["key"] = str(i)
        current_scale["size"] = [
            int(numpy.ceil(previous_scale["size"][0] / 2)),
            int(numpy.ceil(previous_scale["size"][1] / 2)),
            stack_height,
        ]
        current_scale["resolution"] = [
            2 * previous_scale["resolution"][0],
            2 * previous_scale["resolution"][1],
            previous_scale["resolution"][2],
        ]
        info["scales"].append(current_scale)  # type: ignore[attr-defined]

    return info
