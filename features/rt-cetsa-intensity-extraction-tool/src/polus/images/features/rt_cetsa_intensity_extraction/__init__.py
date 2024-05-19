"""RT_CETSA Intensity Extraction Tool."""

import itertools
import logging
import os
import pathlib
import string
from enum import Enum

import bfio
import numpy
import numpy as np
import pandas
from pydantic import BaseModel
from pydantic_core import from_json

logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

ADD_TEMP = True
TEMPERATURE_RANGE = [37, 90]


class IntensityExtractionError(Exception):
    pass


# TODO REMOVE Duplicate : This will cause problems when using plugins together
class PlateSize(Enum):
    SIZE_6 = 6
    SIZE_12 = 12
    SIZE_24 = 24
    SIZE_48 = 48
    SIZE_96 = 96
    SIZE_384 = 384
    SIZE_1536 = 1536


# TODO REMOVE Duplicate : This will cause problems when using plugins together
PLATE_DIMS = {
    PlateSize.SIZE_6: (2, 3),
    PlateSize.SIZE_12: (3, 4),
    PlateSize.SIZE_24: (4, 6),
    PlateSize.SIZE_48: (6, 8),
    PlateSize.SIZE_96: (8, 12),
    PlateSize.SIZE_384: (16, 24),
    PlateSize.SIZE_1536: (32, 48),
}


class PlateParams(BaseModel):
    rotate: int
    """Counterclockwise rotation of image in degrees."""

    bbox: tuple[int, int, int, int]
    """Bounding box of plate after rotation, [ymin,ymax,xmin,xmax]."""

    size: PlateSize
    """The plate size, also determines layout."""

    radius: int
    """Well radius."""

    X: list[int]
    """The the x axis points for wells."""

    Y: list[int]
    """The the y axis points for wells."""


def extract_signal(
    img_paths: list[pathlib.Path],
    mask_path: pathlib.Path,
) -> pandas.DataFrame:
    """Build a DataFrame with well intensity measurements for each temperature.

    Args:
        img_paths: List of image paths.
        mask_path: path to the wells mask.

    Returns:
        Pandas DataFrame.
    """
    if not ADD_TEMP:
        raise NotImplementedError

    num_images = len(img_paths)

    if num_images < 2:
        raise ValueError(
            "provide at least 2 images on the temperature interval "
            + f"({TEMPERATURE_RANGE[0]}-{TEMPERATURE_RANGE[1]})",
        )

    measures: list[tuple[float, list[int]]] = []
    for index, image_path in enumerate(img_paths):
        temp = TEMPERATURE_RANGE[0] + index / (num_images - 1) * (
            TEMPERATURE_RANGE[1] - TEMPERATURE_RANGE[0]
        )
        try:
            row = (temp, extract_wells_intensity_fast(image_path, mask_path))
            measures.append(row)
        except Exception as e:  # noqa
            raise IntensityExtractionError(
                f"could not process image extracted intensity for image : {index+1}/{num_images}",
            ) from e
        logger.info(f"extracted intensity for image : {index+1}/{num_images}")

    # build header
    # check the first plate for number of wells
    nb_wells = len(measures[0][1])
    plate_size = PlateSize(nb_wells)
    plate_row_count = range(PLATE_DIMS[plate_size][0])
    plate_col_count = range(PLATE_DIMS[plate_size][1])
    plate_coords = itertools.product(plate_row_count, plate_col_count)

    header = ["Temperature"] + [
        alphanumeric_row(
            row,
            col,
            (PLATE_DIMS[plate_size][0], PLATE_DIMS[plate_size][1]),
        )
        for row, col in plate_coords
    ]

    # build dataframe
    # roundup temperature
    rows = [[round(measure[0], 1), *measure[1]] for measure in measures]
    df = pandas.DataFrame(rows, columns=header)

    # Set the temperature as the index
    df.set_index("Temperature", inplace=True)

    # Sort the rows by temperature
    df.sort_index(inplace=True)

    return df


def extract_intensity(image: np.ndarray, x: int, y: int, r: int) -> int:
    """Get the well intensity

    Args:
        image: _description_
        x: x-position of the well centerpoint
        y: y-position of the well centerpoint
        r: radius of the well

    Returns:
        int: The background corrected mean well intensity
    """
    assert r >= 5

    # get a large patch to find background pixels
    x_min = max(x - r, 0)
    x_max = min(x + r, image.shape[1])
    y_min = max(y - r, 0)
    y_max = min(y + r, image.shape[0])
    patch = image[y_min:y_max, x_min:x_max]
    background = patch.ravel()
    background.sort()

    # Subtract lowest pixel values from average center pixel values
    return int(np.mean(patch) - np.median(background[: int(0.05 * background.size)]))


def extract_wells_intensity_fast(
    image_path: pathlib.Path,
    mask_path: pathlib.Path,
) -> list[int]:
    """Extract well intensities from RT_CETSA image and mask.

    Args:
        image_path: Path to the RT_CETSA well plate image.
        mask_path: Path to the corresponding params file.

    Returns:
        mean intensity for each wells.
    """
    with bfio.BioReader(image_path) as reader:
        image = reader[:]
    with mask_path.open("r") as f:
        params = PlateParams(**from_json(f.read()))

    intensities = []
    for y, x in itertools.product(range(len(params.Y)), range(len(params.X))):
        intensity = extract_intensity(image, params.X[x], params.Y[y], params.radius)
        print(intensity)
        intensities.append(intensity)

    return intensities


def extract_wells_intensity(
    image_path: pathlib.Path,
    mask_path: pathlib.Path,
) -> list[int]:
    """Extract well intensities from RT_CETSA image and mask.

    Args:
        image_path: Path to the RT_CETSA well plate image.
        mask_path: Path to the corresponding mask image.

    Returns:
        mean intensity for each wells.
    """
    with bfio.BioReader(image_path) as reader:
        image = reader[:]
    with bfio.BioReader(mask_path) as reader:
        mask = reader[:]

    max_mask_index = numpy.max(mask)
    intensities = []

    for mask_label in range(1, max_mask_index + 1):
        # retrieve a bounding box around each well.
        # NOTE this was originally much faster when relying on well positions.
        current_mask = mask == mask_label
        image[current_mask]
        bbox = numpy.argwhere(current_mask)
        bbox_x_min = numpy.min(bbox[0])
        bbox_x_max = numpy.max(bbox[0])
        bbox_y_min = numpy.min(bbox[1])
        bbox_y_max = numpy.max(bbox[1])

        intensity = compute_well_intensity(
            image,
            bbox_y_min,
            bbox_y_max,
            bbox_x_min,
            bbox_x_max,
        )

        intensities.append(intensity)

    return intensities


def compute_well_intensity(image, bbox_y_min, bbox_y_max, bbox_x_min, bbox_x_max):
    # compute corrected intensity
    patch = image[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max]
    background = patch.ravel()
    background.sort()
    # Subtract lowest pixel values from average pixel values
    return int(
        np.mean(patch) - np.mean(background[: int(0.05 * background.size)]),
    )


def alphanumeric_row(row: int, col: int, dims: tuple[int, int]) -> str:
    """Return alphanumeric row:
    For well plate size < 96, coordinates range from A1 to H12
    For well plate size >= 96, coordinates range from A01 to P24
    For well plate size 1536, coordinates range from A01 to AF48
    """
    num_row, num_col = dims
    size = num_row * num_col
    if row >= num_row or col >= num_col:
        msg = f"({row},{col}) out of range for plate size {size}[{num_row},{num_col}]"
        raise ValueError(msg)

    row_alpha = ""
    if row > 26:
        row_alpha = "A"
        row -= 26
    row_alpha = row_alpha + string.ascii_uppercase[row % 26]

    return f"{row_alpha}{col+1:02d}" if size >= 96 else f"{row_alpha}{col+1}"
