"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from os import environ
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from bfio import BioWriter


def get_temp_file(path: Path, suffix: str) -> Path:
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)

@pytest.fixture()
def plugin_dirs(tmp_path: Generator[Path, None, None]) -> tuple[Path, Path]:
    """Create temporary directories used by the plugin."""
    input_dir = tmp_path / "inp_dir"
    output_dir = tmp_path / "out_dir"
    input_dir.mkdir()
    output_dir.mkdir()
    return (input_dir, output_dir)

@pytest.fixture()
def ground_truth_dir(tmp_path: Generator[Path, None, None]) -> Path:
    """Create a temporary directory for storing ground truth."""
    ground_truth_dir = tmp_path / "ground_truth_dir"
    ground_truth_dir.mkdir()
    return ground_truth_dir

@pytest.fixture
def ground_truth_file(ground_truth_dir: Path):
    """Create a ground truth file."""

    image_width = 1024
    image_height = 1024
    image_shape = (image_width, image_height, 1, 1, 1)
    data = np.random(image_shape, dtype=np.uint8)

        # generate the ground truth image
    suffix = environ.get("POLUS_IMG_EXT")
    ground_truth_file = get_temp_file(ground_truth_dir, suffix)
    with BioWriter(ground_truth_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]

@pytest.fixture
def ground_truth():
    return None
