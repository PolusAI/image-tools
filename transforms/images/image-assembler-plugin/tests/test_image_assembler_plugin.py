"""Test image assembler plugin."""

import os
import pathlib
import random
import tempfile

import numpy as np
import pytest
from bfio import BioReader, BioWriter
from src.polus.transforms.images.image_assembler.image_assembler import assemble_image


def get_temp_file(path, suffix):
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)


def create_stitching_vector(filename, offsets):
    """Create stitching vector file."""
    with open(filename, "w") as f:
        for row in offsets:
            for key, value in row.items():
                f.write(f"{key}: {value}; ")
            f.write("\n")


def create_static_stitching_vector(filename):
    """
    Create stitching vector file statically.

    Temporary fix for a bug in filepattern2 where it cannot parse valid variants of stitching vectors.
    """
    offsets = [
        "file: img_r001_c001.ome.tif; corr: -0.0864568939; position: (0, 0); grid: (0, 0);",
        "file: img_r001_c002.ome.tif; corr: -0.657176744; position: (656, 0); grid: (0, 1);",
        "file: img_r002_c001.ome.tif; corr: 0.7119831612; position: (0, 1008); grid: (1, 0);",
        "file: img_r002_c002.ome.tif; corr: 0.2078192665; position: (656, 1008); grid: (1, 1);",
    ]

    with open(filename, "w") as f:
        for row in offsets:
            f.write(f"{row}\n")


@pytest.fixture
def data(tmp_path):
    """Create data fixture."""
    return _data(tmp_path)


def _data(tmp_path):
    """Create data for a basic use case."""
    ground_truth_path = tmp_path / "ground_truth"
    img_path = tmp_path / "img_path"
    stitch_path = tmp_path / "stitch_path"
    out_dir = tmp_path / "out"
    img_path.mkdir()
    stitch_path.mkdir()
    ground_truth_path.mkdir()
    out_dir.mkdir()

    tileSize = 1024
    fovWidth = 1392
    fovHeight = 1040
    offsetX = 1392 - tileSize
    offsetY = 1040 - tileSize
    imageWidth = 2 * tileSize
    imageHeight = 2 * tileSize
    imageShape = (imageWidth, imageHeight, 1, 1, 1)
    data = np.zeros(imageShape, dtype=np.uint8)

    fill_offset = tileSize // 2
    # fmt: off
    data[
        fill_offset: (imageWidth - fill_offset),
        fill_offset: (imageWidth - fill_offset),
    ] = 127
    # fmt: on

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

    suffix = ".ome.tiff"

    ground_truth_file = get_temp_file(ground_truth_path, suffix)

    with BioWriter(ground_truth_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]

    for offset in offsets:
        image_file = img_path / offset["file"]
        offset["corr"] = round(random.uniform(-1, 1), 10)
        originX = offset["position"][0]
        originY = offset["position"][1]
        # fmt: off
        fov = data[originY: (originY + fovHeight), originX: (originX + fovWidth)]
        # fmt: on

        with BioWriter(image_file) as writer:
            writer.X = fovWidth
            writer.Y = fovHeight
            writer[:] = fov

    filename = stitch_path / ("img-global-positions" + ".txt")

    # a bug from filepattern prevents generating from dic for now
    # create_stitching_vector(filename, offsets)
    create_static_stitching_vector(filename)

    return ground_truth_path, img_path, stitch_path, out_dir


def test_image_assembler_plugin(data):
    """Test correctness of image assembler plugin in a basic case."""
    ground_truth_path, img_path, stitch_path, out_dir = data

    assemble_image(img_path, stitch_path, out_dir, False)

    assert len(os.listdir(ground_truth_path)) == 1
    assert len(os.listdir(out_dir)) == 1

    ground_truth_file = os.listdir(ground_truth_path)[0]
    assembled_image_file = os.listdir(out_dir)[0]
    gt = ground_truth_path / ground_truth_file
    out = out_dir / assembled_image_file

    with BioReader(gt) as gt_reader:
        with BioReader(out) as out_reader:
            assert gt_reader.shape == out_reader.shape
            assert (gt_reader[:] == out_reader[:]).all()


"""Debugging hook"""
if __name__ == "__main__":
    """Test image assembler plugin."""
    with tempfile.TemporaryDirectory() as tmpDir:
        # tmpPath = pathlib.Path(tempfile.mkdtemp()) # keep data around for further debugging
        tmpPath = pathlib.Path(tmpDir)
        test_image_assembler_plugin(_data(tmpPath))
