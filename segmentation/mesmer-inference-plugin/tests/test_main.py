"""Mesmer Inference."""
import os
import pathlib
import shutil
import tempfile
from collections.abc import Generator

import filepattern
import numpy as np
import pytest
import skimage
from bfio import BioReader
from deepcell.applications import Mesmer
from polus.plugins.segmentation.mesmer_inference.__main__ import app
from polus.plugins.segmentation.mesmer_inference.padded import get_data
from polus.plugins.segmentation.mesmer_inference.padded import padding
from polus.plugins.segmentation.mesmer_inference.padded import run
from skimage import io
from typer.testing import CliRunner

runner = CliRunner()
mess_app = Mesmer()


@pytest.fixture()
def model_dir() -> Generator[pathlib.Path, None, None]:
    """Model directory for saving intensity images."""
    MODEL_DIR = os.path.expanduser(os.path.join("~", ".keras", "models"))
    return pathlib.Path(MODEL_DIR, "MultiplexSegmentation")


@pytest.fixture()
def inp_dir() -> Generator[str, None, None]:
    """Create directory for saving intensity images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Create output directory."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture()
def synthetic_images(inp_dir) -> Generator[pathlib.Path, None, None]:
    """Generate random synthetic images."""
    for i in range(5):
        image = skimage.data.coins()
        ch0_name = f"y{i}_r{i}_c0.tif"
        ch1_name = f"y{i}_r{i}_c1.tif"
        gtch0 = pathlib.Path(inp_dir, ch0_name)
        gtch1 = pathlib.Path(inp_dir, ch1_name)
        io.imsave(gtch0, image)
        io.imsave(gtch1, image)

    return inp_dir


@pytest.fixture(
    params=[
        (
            "y{y+}_r{r+}_c0.tif",
            "y{y+}_r{r+}_c1.tif",
            512,
            2,
            "mesmerNuclear",
            ".ome.tif",
        ),
        (
            "y{y+}_r{r+}_c0.tif",
            "y{y+}_r{r+}_c0.tif",
            512,
            2,
            "mesmerNuclear",
            ".ome.zarr",
        ),
        (
            "y{y+}_r{r+}_c0.tif",
            "y{y+}_r{r+}_c1.tif",
            512,
            2,
            "mesmerWholeCell",
            ".ome.tif",
        ),
        ("y{y+}_r{r+}_c0.tif", None, 512, 1, "nuclear", ".ome.zarr"),
        ("y{y+}_r{r+}_c1.tif", None, 512, 1, "cytoplasm", ".ome.zarr"),
    ],
)
def get_params(request):
    """To get the parameter of the fixture."""
    return request.param


def test_get_data(synthetic_images, get_params) -> None:
    """Test prepare intensity images for prediction.

    This unit test runs the get_data and validates the outputs
    """
    pattern1, pattern2, size, channel, model, _ = get_params
    gt_dir = synthetic_images

    result = get_data(
        inp_dir=gt_dir,
        file_pattern_1=pattern1,
        file_pattern_2=pattern2,
        size=size,
        model=model,
    )
    assert len(result) == (len(os.listdir(synthetic_images))) / 2

    assert result[0].shape == (size, size, channel)


def test_padding(synthetic_images) -> None:
    """Test image padding.

    This unit test runs the padding and validates the output image is padded correctly
    """
    file_pattern_1 = "y{y:d}_r{r:d}_c0.tif"
    inp_dir = synthetic_images
    fp = filepattern.FilePattern(inp_dir, file_pattern_1)
    for pad_size in [512, 1024, 2048]:
        for file in fp():
            br = BioReader(file[1][0])
            image = br.read()
            y, x = image.shape
            pad_img, pad_dims = padding(image, y, x, True, size=pad_size)

            assert pad_img.shape == (pad_size, pad_size)
            assert len(pad_dims) == 4


def test_run(synthetic_images, model_dir, get_params, output_directory) -> None:
    """Test mesmer predicted outputs.

    This unit test runs the run and validates the predictions are saved correctly
    """
    pattern1, pattern2, size, _, model, fileext = get_params
    inp_dir = synthetic_images
    model_path = model_dir
    run(inp_dir, size, model_path, pattern1, pattern2, model, fileext, output_directory)

    for f in pathlib.Path(output_directory).iterdir():
        br = BioReader(f)
        image = br.read()
        assert image.dtype == np.uint16


def test_cli(synthetic_images, model_dir, output_directory, get_params) -> None:
    """Test Cli."""
    pattern1, pattern2, size, _, model, fileext = get_params
    inp_dir = synthetic_images
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
