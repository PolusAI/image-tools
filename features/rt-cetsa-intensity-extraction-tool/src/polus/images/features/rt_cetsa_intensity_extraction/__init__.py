"""RT_CETSA Intensity Extraction Tool."""
__version__ = "0.4.0-dev0"

import itertools
import logging
import os
import pathlib
import string

import bfio
import filepattern
import numpy
import numpy as np
import pandas
from polus.images.segmentation.rt_cetsa_plate_extraction.core import PLATE_DIMS
from polus.images.segmentation.rt_cetsa_plate_extraction.core import PlateParams
from polus.images.segmentation.rt_cetsa_plate_extraction.core import PlateSize
from pydantic_core import from_json

logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


class IntensityExtractionError(Exception):
    """Raise if an image could not be processed."""

    pass


def sort_and_extract_signal(
    img_dir: pathlib.Path,
    plate_params: pathlib.Path,
    file_pattern: str,
):
    """Build a DataFrame with well intensity measurements for each temperature.

    This is the top level method for this module and should be called by client's code.
    In addition to extracting signal,
    it sort images provided in the path provided according to file pattern.
    For convenience, a default pattern using the existing RT Cetsa naming scheme is provided.

    Args:
        img_dir: path to the image input directory.
        file_pattern: filepattern used to sort the image.
        mask_path: path to the plate params file.
        temp_interval: temperature range on which the img_dir are collected.
            we assume a linear temperature increase to build the result dataframe.

    Returns:
        Pandas DataFrame.
    """
    fps = sort_fps(img_dir, file_pattern)
    return extract_signal(fps, plate_params)


def sort_fps(img_dir: pathlib.Path, file_pattern: str):
    """Sort image with the provided filepattern.

    If multiple indexing variables are provided, we only index using the first one.
    """
    print("img_dir", img_dir)
    print("file_pattern", file_pattern)

    fps = filepattern.FilePattern(img_dir, file_pattern)

    if len(fps.get_variables()) == 0:
        msg = "A filepattern with one indexing variable is needed to sort the input images."
        raise ValueError(
            msg,
        )

    if "index" not in fps.get_variables() or "temp" not in fps.get_variables():
        msg = "filepattern should contain at least two variables : index and temp"
        raise ValueError(
            msg,
        )

    return fps()


def extract_signal(fps, plate_params: pathlib.Path) -> pandas.DataFrame:
    """Build a DataFrame with well intensity measurements for each temperature.

    Args:
        fp: filepatterns
        mask_path: path to the wells mask.

    Returns:
        Pandas DataFrame.
    """
    measures: list[tuple[float, list[int]]] = []
    num_images = len(fps)
    for index, fp in enumerate(fps):
        image_path = fp[1][0]
        temp = fp[0]["temp"]
        try:
            row = (temp, extract_wells_intensity(image_path, plate_params))
            measures.append(row)
        except Exception as e:  # noqa
            raise IntensityExtractionError(
                f"could not process image extracted intensity for image : {index+1}/{num_images}",
            ) from e
        logger.info(f"extracted intensity for image : {index+1}/{num_images}")

    # build header
    # check the first plate for the plate dimensions
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
    rows = [[round(measure[0], 2), *measure[1]] for measure in measures]
    df = pandas.DataFrame(rows, columns=header)

    # Set the temperature as the index
    df.set_index("Temperature", inplace=True)

    # Sort the rows by temperature
    df.sort_index(inplace=True)

    return df


def extract_wells_intensity(
    image_path: pathlib.Path,
    params_path: pathlib.Path,
) -> list[int]:
    """Extract well intensities from RT_CETSA image and mask.

    Args:
        image_path: Path to the RT_CETSA well plate image.
        params_path: Path to the corresponding params file.

    Returns:
        corrected intensity for each well.
    """
    with bfio.BioReader(image_path) as reader:
        image = reader[:]
    with params_path.open("r") as f:
        params = PlateParams(**from_json(f.read()))

    intensities = []

    for y, x in itertools.product(range(len(params.Y)), range(len(params.X))):
        intensity = extract_intensity(
            image,
            params.X[x],
            params.Y[y],
            params.roi_radius,
        )
        intensities.append(intensity)

    return intensities


def extract_intensity(image: np.ndarray, x: int, y: int, r: int) -> int:
    """Get the well intensity

    Args:
        image: _description_
        x: x-position of the well centerpoint
        y: y-position of the well centerpoint
        r: radius of the circle inscribed in the square area of interest.

    Returns:
        int: The background corrected mean well intensity
    """
    # we take a square area around the well center
    x_min = max(x - r, 0)
    x_max = min(x + r, image.shape[1])
    y_min = max(y - r, 0)
    y_max = min(y + r, image.shape[0])

    patch = image[y_min:y_max, x_min:x_max]
    background = patch.ravel()
    background.sort()

    # Subtract lowest pixel values from average patch pixel values
    return int(np.mean(patch) - np.median(background[: int(0.05 * background.size)]))


def extract_wells_intensity_from_mask(
    image_path: pathlib.Path,
    mask_path: pathlib.Path,
) -> list[int]:
    """Extract well intensities from RT_CETSA images using a labeled mask.

    This method is degree of magnitude slower than extract_wells_intensity
    and is just provided for convenience. Consider using extract_wells_intensity instead.

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


def compute_well_intensity(
    image: np.ndarray,
    bbox_y_min: int,
    bbox_y_max: int,
    bbox_x_min: int,
    bbox_x_max: int,
):
    """Compute corrected intensity."""
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
