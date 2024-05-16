"""RT_CETSA Intensity Extraction Tool."""

__version__ = "0.1.0"

import itertools
import pathlib
import string
from enum import Enum

import bfio
import numpy
import pandas
from skimage.draw import disk
from skimage.transform import rotate

TEMPERATURE_RANGE = [37, 95]


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
) -> list[float]:
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

    mean_background = numpy.mean(image[mask == 0])

    max_mask_index = numpy.max(mask)
    intensities = []
    for i in range(1, max_mask_index + 1):
        mean_mask_intensities = numpy.mean(image[mask == i])
        intensities.append(mean_mask_intensities - mean_background)

    return intensities


def index_to_battleship(x: int, y: int, size: PlateSize) -> str:
    """Get the battleship notation of a well index.

    Args:
        x: x-position of the well centerpoint
        y: y-position of the well centerpoint

    Returns:
        str: The string representation of the well index (i.e. A1)
    """
    # The y-position should be converted to an uppercase well letter
    row = ""
    if y >= 26:
        row = "A"
    row = row + string.ascii_uppercase[y % 26]

    return f"{row}{x+1}"


def build_df(
    inp_paths: list[pathlib.Path],
    mask_path: pathlib.Path,
) -> pandas.DataFrame:
    """Build a DataFrame with well intensities.

    Args:
        file_paths: List of tuples with image and mask paths.

    Returns:
        Pandas DataFrame with well intensities.
    """
    intensities: list[tuple[float, list[float]]] = []
    for i, f in enumerate(inp_paths):
        temp = TEMPERATURE_RANGE[0] + i / (len(inp_paths) - 1) * (
            TEMPERATURE_RANGE[1] - TEMPERATURE_RANGE[0]
        )
        intensities.append((temp, extract_intensities(f, mask_path)))

    # sort intensities by temperature
    intensities.sort(key=lambda x: x[0])
    rows: list[list[float]] = [[a, *b] for a, b in intensities]

    # build header
    header = ["Temperature"]
    plate_size = PlateSize(len(intensities[0][1]))

    for x, y in itertools.product(
        range(PLATE_DIMS[plate_size][0]),
        range(PLATE_DIMS[plate_size][1]),
    ):
        header.append(index_to_battleship(x, y, plate_size))

    # build DataFrame
    df = pandas.DataFrame(rows, columns=header)

    # Set the temperature as the index
    df.set_index("Temperature", inplace=True)

    # Sort the rows by temperature
    df.sort_index(inplace=True)

    return df
