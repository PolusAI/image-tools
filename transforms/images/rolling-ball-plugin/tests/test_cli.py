from typer.testing import CliRunner
import tempfile
import pytest
from src.polus.plugins.transforms.images.rolling_ball.__main__ import app as app
import numpy
from bfio import BioReader, BioWriter
import logging
import json
import faulthandler

faulthandler.enable()

def get_temp_file(path, suffix):
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
def image():
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
def infile(image, paths):
    """Create an image file in the input directory."""
    inp_dir, _ = paths
    infile = get_temp_file(inp_dir, "ome.tiff")
    
    with BioWriter(infile) as writer:
        writer.X = image.shape[0]
        writer.Y = image.shape[1]
        writer[:] = image[:]

    return infile


def test_cli(infile, paths):  # noqa
    """Test the Rolling Ball plugin from the command line."""
    runner = CliRunner()

    inp_dir, out_dir = paths

    result = runner.invoke(
        app,
        [
            "--inputDir",
            str(inp_dir),
            "--outputDir",
            str(out_dir)
        ],
    )

    assert result.exit_code == 0

def test_cli_preview(infile, paths, caplog):  # noqa
    """Test the client with the preview option."""

    runner = CliRunner()

    inp_dir, out_dir = paths

    result = runner.invoke(
        app,
        [
            "--inputDir",
            str(inp_dir),
            "--outputDir",
            str(out_dir),
            "--preview"
        ],
    )

    # no error
    assert result.exit_code == 0

    with open(out_dir / "preview.json" ) as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    result = plugin_json["outputDir"]
    assert len(result) == 1
    assert result[0] == infile.name

def test_cli_bad_input(paths):  # noqa
    """Test the Rolling Ball plugin from the command line."""
    
    runner = CliRunner()

    inp_dir, out_dir = paths
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
        [
            "--inputDir",
            str(inp_dir),
            "--outputDir",
            str(out_dir)
        ],
    )

    assert result.exc_info[0] is ValueError
