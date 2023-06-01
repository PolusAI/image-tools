import tempfile
from pathlib import Path
import numpy
import pytest
from bfio import BioWriter

def get_temp_file(path: Path, suffix: str):
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)


@pytest.fixture()
def paths(tmp_path):
    """Create temporary directories"""
    input_dir = tmp_path / "inp"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    return (input_dir, output_dir)


@pytest.fixture()
def image_data():
    """Create a numpy array to generate an input image."""
    image_size = 64
    image_shape = (image_size, image_size)
    random_image = numpy.random.randint(
        low=0,
        high=255,
        size=image_shape,
        dtype=numpy.uint8,
    )
    return random_image


@pytest.fixture()
def image_file(image_data, paths):
    """Create an image file in the input directory."""
    inp_dir, _ = paths
    infile = get_temp_file(inp_dir, "ome.tiff")

    with BioWriter(infile) as writer:
        writer.X = image_data.shape[0]
        writer.Y = image_data.shape[1]
        writer[:] = image_data[:]

    return infile

@pytest.fixture()
def output_file(paths):
    _, output_dir = paths
    return get_temp_file(output_dir, "ome.tiff")

@pytest.fixture()
def options():
    return {
        "ball_radius": 25,
        "light_background": False
    }
