"""RT_CETSA Intensity Extraction Tool."""

__version__ = "0.1.0"

import itertools
import pathlib
import string
from enum import Enum

import bfio
import numpy
import numpy as np
import pandas
from skimage.draw import disk
from skimage.transform import rotate

ADD_TEMP = True
TEMPERATURE_RANGE = [37, 90]


class PlateSize(Enum):
    SIZE_6 = 6
    SIZE_12 = 12
    SIZE_24 = 24
    SIZE_48 = 48
    SIZE_96 = 96
    SIZE_384 = 384
    SIZE_1536 = 1536


PLATE_DIMS = {
    PlateSize.SIZE_6: (2, 3),
    PlateSize.SIZE_12: (3, 4),
    PlateSize.SIZE_24: (4, 6),
    PlateSize.SIZE_48: (6, 8),
    PlateSize.SIZE_96: (9, 12),
    PlateSize.SIZE_384: (16, 24),
    PlateSize.SIZE_1536: (32, 48),
}


def extract_intensities(
    image_path: pathlib.Path,
    mask_path: pathlib.Path,
) -> list[int]:
    """Extract well intensities from RT_CETSA image and mask.

    Args:
        image_path: Path to the RT_CETSA image.
        mask_path: Path to the mask image.

    Returns:
        Pandas DataFrame with well intensities.
    """
    with bfio.BioReader(image_path) as reader:
        image = reader[:]
    with bfio.BioReader(mask_path) as reader:
        mask = reader[:]

    max_mask_index = numpy.max(mask)
    intensities = []
    for i in range(1, max_mask_index + 1):
        current_mask = mask == i
        image[current_mask]

        # find a square bounding box around the mask
        bbox = numpy.argwhere(current_mask)
        bbox_x_min = numpy.min(bbox[0])
        bbox_x_max = numpy.max(bbox[0])
        bbox_y_min = numpy.min(bbox[1])
        bbox_y_max = numpy.max(bbox[1])

        patch = image[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max]
        background = patch.ravel()
        background.sort()
        corrected_mean_intensity = int(
            np.mean(patch) - np.mean(background[: int(0.05 * background.size)]),
        )

        # Subtract lowest pixel values from average pixel values
        intensities.append(corrected_mean_intensity)

    return intensities


def index_to_battleship(x: int, y: int, size: PlateSize) -> str:
    """Get the battleship notation of a well index.

    Args:
        x: x-position of the well centerpoint
        y: y-position of the well centerpoint
        size: size of the plate

    Returns:
        str: The string representation of the well index (i.e. A1)
    """
    # The y-position should be converted to an uppercase well letter
    row = ""
    if y >= 26:
        row = "A"
    row = row + string.ascii_uppercase[y % 26]

    return f"{row}{x + 1:02d}" if size.value >= 96 else f"{row}{x + 1}"


def build_df(
    file_paths: list[tuple[pathlib.Path, pathlib.Path]],
) -> pandas.DataFrame:
    """Build a DataFrame with well intensities.

    Args:
        file_paths: List of tuples with image and mask paths.

    Returns:
        Pandas DataFrame with well intensities.
    """
    intensities: list[tuple[float, list[int]]] = []

    if not ADD_TEMP:
        raise NotImplementedError

    if len(file_paths) < 1:
        raise ValueError(
            "provide at least 2 images on the temperature interval"
            + f"{TEMPERATURE_RANGE[0]}-{TEMPERATURE_RANGE[1]}",
        )

    for i, (image_path, mask_path) in enumerate(file_paths):
        temp = TEMPERATURE_RANGE[0] + i / (len(file_paths) - 1) * (
            TEMPERATURE_RANGE[1] - TEMPERATURE_RANGE[0]
        )
        row = (temp, extract_intensities(image_path, mask_path))
        intensities.append(row)

    # sort intensities by temperature
    intensities.sort(key=lambda x: x[0])

    # check the first plate for number of wells
    nb_wells = len(intensities[0][1])
    plate_size = PlateSize(nb_wells)

    # build header
    header = ["Temperature"]
    plate_row = range(PLATE_DIMS[plate_size][0])
    plate_col = range(PLATE_DIMS[plate_size][1])

    for y, x in itertools.product(plate_row, plate_col):
        header.append(index_to_battleship(x, y, plate_size))

    # build DataFrame
    rows = [[round(measure[0], 1), *measure[1]] for measure in intensities]
    df = pandas.DataFrame(rows, columns=header)

    # Set the temperature as the index
    df.set_index("Temperature", inplace=True)

    # Sort the roxws by temperature
    df.sort_index(inplace=True)

    return df
