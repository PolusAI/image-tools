<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 254a68a (update : update to new standards.)
"""Test fixtures.

Set up all data used in tests.
"""
<<<<<<< HEAD

import random
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from bfio import BioWriter


def get_temp_file(path: Path, suffix: str) -> Path:
=======
import tempfile
import pytest
=======
>>>>>>> 254a68a (update : update to new standards.)

import random
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from bfio import BioWriter

<<<<<<< HEAD
def get_temp_file(path: Path, suffix: str):
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======

def get_temp_file(path: Path, suffix: str) -> Path:
>>>>>>> 254a68a (update : update to new standards.)
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)

<<<<<<< HEAD
<<<<<<< HEAD

@pytest.fixture()
def plugin_dirs(tmp_path: Generator[Path, None, None]) -> tuple[Path, Path, Path]:
    """Create temporary directories."""
=======
@pytest.fixture()
def plugin_dirs(tmp_path):
    """Create temporary directories"""
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======

@pytest.fixture()
def plugin_dirs(tmp_path: Generator[Path, None, None]) -> tuple[Path, Path, Path]:
    """Create temporary directories."""
>>>>>>> 254a68a (update : update to new standards.)
    input_dir = tmp_path / "inp_dir"
    output_dir = tmp_path / "out_dir"
    stitch_dir = tmp_path / "stitch_dir"
    input_dir.mkdir()
    output_dir.mkdir()
    stitch_dir.mkdir()
    return (input_dir, stitch_dir, output_dir)

<<<<<<< HEAD
<<<<<<< HEAD

@pytest.fixture()
def ground_truth_dir(tmp_path: Generator[Path, None, None]) -> Path:
    """Create temporary directories."""
=======
@pytest.fixture()
def ground_truth_dir(tmp_path):
    """Create temporary directories"""
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======

@pytest.fixture()
def ground_truth_dir(tmp_path: Generator[Path, None, None]) -> Path:
    """Create temporary directories."""
>>>>>>> 254a68a (update : update to new standards.)
    ground_truth_dir = tmp_path / "ground_truth_dir"
    ground_truth_dir.mkdir()
    return ground_truth_dir

<<<<<<< HEAD
<<<<<<< HEAD

@pytest.fixture
def data(plugin_dirs: tuple[Path, Path, Path], ground_truth_dir: Path) -> None:
    """Generate test data.

    Create a grounth truth image
    Create a stitching vector
    Create a set of partial images that will be assembled
=======
=======

>>>>>>> 254a68a (update : update to new standards.)
@pytest.fixture
def data(plugin_dirs: tuple[Path, Path, Path], ground_truth_dir: Path) -> None:
    """Generate test data.

<<<<<<< HEAD
        Create a grounth truth image
        Create a stitching vector
        Create a set of partial images that will be assembled
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
    Create a grounth truth image
    Create a stitching vector
    Create a set of partial images that will be assembled
>>>>>>> 254a68a (update : update to new standards.)
    """
    img_path, stitch_path, _ = plugin_dirs
    ground_truth_path = ground_truth_dir

    # generate the image data
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 254a68a (update : update to new standards.)
    tile_size = 1024
    fov_width = 1392
    fov_height = 1040
    offset_x = 1392 - tile_size
    offset_y = 1040 - tile_size
    image_width = 2 * tile_size
    image_height = 2 * tile_size
    image_shape = (image_width, image_height, 1, 1, 1)
    data = np.zeros(image_shape, dtype=np.uint8)

    # max value for np.uint8 so we have a white square in the middle of the image
    fill_value = 127
    fill_offset = tile_size // 2
<<<<<<< HEAD
    # fmt: off
    data[
        fill_offset: (image_width - fill_offset),
        fill_offset: (image_width - fill_offset),
    ] = fill_value
=======
    tileSize = 1024
    fovWidth = 1392
    fovHeight = 1040
    offsetX = 1392 - tileSize
    offsetY = 1040 - tileSize
    imageWidth = 2 * tileSize
    imageHeight = 2 * tileSize
    imageShape = (imageWidth, imageHeight, 1, 1, 1)
    data = np.zeros(imageShape, dtype=np.uint8)
    
    #max value for np.uint8 so we have a white square in the middle of the image
    FILL_VALUE = 127
    fill_offset = tileSize // 2
    # fmt: off
    data[
        fill_offset: (imageWidth - fill_offset),
        fill_offset: (imageWidth - fill_offset),
    ] = FILL_VALUE 
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
    # fmt: off
    data[
        fill_offset: (image_width - fill_offset),
        fill_offset: (image_width - fill_offset),
    ] = fill_value
>>>>>>> 254a68a (update : update to new standards.)
    # fmt: on

    # generate the ground truth image
    suffix = ".ome.tiff"
    ground_truth_file = get_temp_file(ground_truth_path, suffix)
    with BioWriter(ground_truth_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]

    stitching_vector = stitch_path / ("img-global-positions" + ".txt")

    # stitching data
    offsets = [
        {"grid": (0, 0), "file": "img_r001_c001.ome.tif", "position": (0, 0)},
        {
            "grid": (0, 1),
            "file": "img_r001_c002.ome.tif",
<<<<<<< HEAD
<<<<<<< HEAD
            "position": (tile_size - offset_x, 0),
=======
            "position": (tileSize - offsetX, 0),
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
            "position": (tile_size - offset_x, 0),
>>>>>>> 254a68a (update : update to new standards.)
        },
        {
            "grid": (1, 0),
            "file": "img_r002_c001.ome.tif",
<<<<<<< HEAD
<<<<<<< HEAD
            "position": (0, tile_size - offset_y),
=======
            "position": (0, tileSize - offsetY),
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
            "position": (0, tile_size - offset_y),
>>>>>>> 254a68a (update : update to new standards.)
        },
        {
            "grid": (1, 1),
            "file": "img_r002_c002.ome.tif",
<<<<<<< HEAD
<<<<<<< HEAD
            "position": (tile_size - offset_x, tile_size - offset_y),
=======
            "position": (tileSize - offsetX, tileSize - offsetY),
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
            "position": (tile_size - offset_x, tile_size - offset_y),
>>>>>>> 254a68a (update : update to new standards.)
        },
    ]
    for offset in offsets:
        offset["corr"] = round(random.uniform(-1, 1), 10)

    # create stitching vector
    # TODO CHECK Filepattern updates.
    # A bug from filepattern prevents generating from dic for now
    stitching_data = [
        "file: img_r001_c001.ome.tif; corr: -0.0864568939; position: (0, 0); grid: (0, 0);",
        "file: img_r001_c002.ome.tif; corr: -0.657176744; position: (656, 0); grid: (0, 1);",
        "file: img_r002_c001.ome.tif; corr: 0.7119831612; position: (0, 1008); grid: (1, 0);",
        "file: img_r002_c002.ome.tif; corr: 0.2078192665; position: (656, 1008); grid: (1, 1);",
    ]

<<<<<<< HEAD
<<<<<<< HEAD
    with Path.open(stitching_vector, "w") as f:
=======
    with open(stitching_vector, "w") as f:
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
    with Path.open(stitching_vector, "w") as f:
>>>>>>> 254a68a (update : update to new standards.)
        for row in stitching_data:
            f.write(f"{row}\n")

    # TODO When Filepattern is fixed, generate the stitching vector like that
    # # create the stitching vector
    # with open(filename, "w") as f:
    #     for row in offsets:
    #         for key, value in row.items():
    #             f.write(f"{key}: {value}; ")
    #         f.write("\n")

<<<<<<< HEAD
<<<<<<< HEAD
    # generate the partial images
    for offset in offsets:
        image_file = img_path / offset["file"]
        origin_x = offset["position"][0]
        origin_y = offset["position"][1]
        # fmt: off
        fov = data[origin_y: (origin_y + fov_height), origin_x: (origin_x + fov_width)]
        # fmt: on
        with BioWriter(image_file) as writer:
            writer.X = fov_width
            writer.Y = fov_height
            writer[:] = fov
=======

=======
>>>>>>> 254a68a (update : update to new standards.)
    # generate the partial images
    for offset in offsets:
        image_file = img_path / offset["file"]
        origin_x = offset["position"][0]
        origin_y = offset["position"][1]
        # fmt: off
        fov = data[origin_y: (origin_y + fov_height), origin_x: (origin_x + fov_width)]
        # fmt: on
        with BioWriter(image_file) as writer:
<<<<<<< HEAD
            writer.X = fovWidth
            writer.Y = fovHeight
            writer[:] = fov
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
            writer.X = fov_width
            writer.Y = fov_height
            writer[:] = fov
>>>>>>> 254a68a (update : update to new standards.)
