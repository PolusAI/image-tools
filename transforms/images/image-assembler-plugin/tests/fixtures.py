import tempfile
import pytest

from pathlib import Path
import numpy as np
import random

from bfio import BioWriter

def get_temp_file(path: Path, suffix: str):
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)

@pytest.fixture()
def plugin_dirs(tmp_path):
    """Create temporary directories"""
    input_dir = tmp_path / "inp_dir"
    output_dir = tmp_path / "out_dir"
    stitch_dir = tmp_path / "stitch_dir"
    input_dir.mkdir()
    output_dir.mkdir()
    stitch_dir.mkdir()
    return (input_dir, stitch_dir, output_dir)

@pytest.fixture()
def ground_truth_dir(tmp_path):
    """Create temporary directories"""
    ground_truth_dir = tmp_path / "ground_truth_dir"
    ground_truth_dir.mkdir()
    return ground_truth_dir

@pytest.fixture
def data(plugin_dirs, ground_truth_dir):
    """Generate test data.

        Create a grounth truth image
        Create a stitching vector
        Create a set of partial images that will be assembled
    """
    img_path, stitch_path, _ = plugin_dirs
    ground_truth_path = ground_truth_dir

    # generate the image data
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
            "position": (tileSize - offsetX, 0),
        },
        {
            "grid": (1, 0),
            "file": "img_r002_c001.ome.tif",
            "position": (0, tileSize - offsetY),
        },
        {
            "grid": (1, 1),
            "file": "img_r002_c002.ome.tif",
            "position": (tileSize - offsetX, tileSize - offsetY),
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

    with open(stitching_vector, "w") as f:
        for row in stitching_data:
            f.write(f"{row}\n")

    # TODO When Filepattern is fixed, generate the stitching vector like that
    # # create the stitching vector
    # with open(filename, "w") as f:
    #     for row in offsets:
    #         for key, value in row.items():
    #             f.write(f"{key}: {value}; ")
    #         f.write("\n")


    # generate the partial images
    for offset in offsets:
        image_file = img_path / offset["file"]
        originX = offset["position"][0]
        originY = offset["position"][1]
        # fmt: off
        fov = data[originY: (originY + fovHeight), originX: (originX + fovWidth)]
        # fmt: on
        with BioWriter(image_file) as writer:
            writer.X = fovWidth
            writer.Y = fovHeight
            writer[:] = fov