"""Mesmer Inference."""

import os
import pathlib

import filepattern
import numpy as np
import pytest
from bfio import BioReader
from deepcell.applications import Mesmer
from polus.plugins.segmentation.mesmer_inference.__main__ import app
from polus.plugins.segmentation.mesmer_inference.padded import get_data
from polus.plugins.segmentation.mesmer_inference.padded import padding
from polus.plugins.segmentation.mesmer_inference.padded import run
from typer.testing import CliRunner

from .conftest import DIR_RETURN_TYPE
from .conftest import clean_directories

runner = CliRunner()
mess_app = Mesmer()


@pytest.mark.skipif("not config.getoption('slow')")
def test_get_data(
    synthetic_images: DIR_RETURN_TYPE,
    get_scaled_params: pytest.FixtureRequest,
) -> None:
    """Test prepare intensity images for prediction.

    This unit test runs the get_data and validates the outputs
    """
    pattern1, pattern2, size, channel, model, _ = get_scaled_params
    gt_dir, _ = synthetic_images

    result = get_data(
        inp_dir=gt_dir,
        file_pattern_1=pattern1,
        file_pattern_2=pattern2,
        size=size,
        model=model,
    )
    assert len(result) == (len(os.listdir(gt_dir))) / 2

    assert result[0].shape == (size, size, channel)
    clean_directories()


@pytest.mark.skipif("not config.getoption('slow')")
def test_padding(synthetic_images: DIR_RETURN_TYPE) -> None:
    """Test image padding.

    This unit test runs the padding and validates the output image is padded correctly
    """
    file_pattern_1 = "y{y:d}_r{r:d}_c0.tif"
    inp_dir, _ = synthetic_images
    fp = filepattern.FilePattern(inp_dir, file_pattern_1)
    for pad_size in [512, 1024, 2048]:
        for file in fp():
            br = BioReader(file[1][0])
            image = br.read()
            y, x = image.shape
            pad_img, pad_dims = padding(image, y, x, True, size=pad_size)

            assert pad_img.shape == (pad_size, pad_size)
            assert len(pad_dims) == 4
    clean_directories()


@pytest.mark.skipif("not config.getoption('slow')")
def test_run(
    synthetic_images: DIR_RETURN_TYPE,
    model_dir: pathlib.Path,
    get_scaled_params: pytest.FixtureRequest,
) -> None:
    """Test mesmer predicted outputs.

    This unit test runs the run and validates the predictions are saved correctly
    """
    pattern1, pattern2, size, _, model, fileext = get_scaled_params
    inp_dir, output_directory = synthetic_images
    model_path = model_dir
    run(inp_dir, size, model_path, pattern1, pattern2, model, fileext, output_directory)

    for f in pathlib.Path(output_directory).iterdir():
        br = BioReader(f)
        image = br.read()
        assert image.dtype == np.uint16

    clean_directories()


def test_cli(
    synthetic_images: DIR_RETURN_TYPE,
    model_dir: pathlib.Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test Cli."""
    pattern1, pattern2, size, _, model, fileext = get_params
    inp_dir, output_directory = synthetic_images
    model_path = model_dir

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--tileSize",
            size,
            "--modelPath",
            model_path,
            "--filePatternTest",
            pattern1,
            "--filePatternWholeCell",
            pattern2,
            "--fileExtension",
            fileext,
            "--model",
            model,
            "--fileExtension",
            fileext,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()
