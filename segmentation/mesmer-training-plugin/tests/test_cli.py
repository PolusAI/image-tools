"""Mesmer Training."""

import pathlib
import shutil
import tempfile

import numpy as np
import pytest
from polus.plugins.segmentation.mesmer_training.__main__ import app
from skimage import filters
from skimage import io
from skimage import measure
from typer.testing import CliRunner

DIR_RETURN_TYPE = tuple[
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
]

runner = CliRunner()


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in pathlib.Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("data_"):
            shutil.rmtree(d)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


@pytest.fixture()
def synthetic_images() -> DIR_RETURN_TYPE:
    """Generate random synthetic train images."""
    directory = pathlib.Path(tempfile.mkdtemp(prefix="data_", dir=pathlib.Path.cwd()))
    training_images = directory.joinpath("training_images")
    training_images.mkdir(parents=True, exist_ok=True)
    training_labels = directory.joinpath("training_labels")
    training_labels.mkdir(parents=True, exist_ok=True)
    test_images = directory.joinpath("test_images")
    test_images.mkdir(parents=True, exist_ok=True)
    test_labels = directory.joinpath("test_labels")
    test_labels.mkdir(parents=True, exist_ok=True)
    out_dir = directory.joinpath("outdir")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 50
    size = 256
    for i in range(3):
        image = np.zeros((size, size))
        points = size * np.random.random((2, n**1))
        image[(points[0]).astype(int), (points[1]).astype(int)] = 1
        image = filters.gaussian(image, sigma=size / (7.0 * n))
        train_int = pathlib.Path(training_images, f"y{i}_r{i}_c0.tif")
        train_label = pathlib.Path(training_labels, f"y{i}_r{i}_c0.tif")
        test_int = pathlib.Path(test_images, f"y{i}_r{i}_c0.tif")
        test_label = pathlib.Path(test_labels, f"y{i}_r{i}_c0.tif")
        blobs = image > image.mean()
        blobs_labels = measure.label(blobs, background=0)
        io.imsave(train_int, image)
        io.imsave(train_label, blobs_labels)
        io.imsave(test_int, image)
        io.imsave(test_label, blobs_labels)
    return (
        pathlib.Path(training_images),
        pathlib.Path(training_labels),
        pathlib.Path(test_images),
        pathlib.Path(test_labels),
        out_dir,
    )


SHORT_PRAMS = [
    (
        "vgg19",
        3,
    ),
    (
        "vgg16",
        2,
    ),
    (
        "resnet50v2",
        2,
    ),
]


@pytest.fixture(params=SHORT_PRAMS)
def short_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


def test_cli(
    synthetic_images: DIR_RETURN_TYPE,
    short_params: pytest.FixtureRequest,
) -> None:
    """Test Cli."""
    model_backbone, batch_size = short_params
    (
        training_images,
        training_labels,
        test_images,
        test_labels,
        output_directory,
    ) = synthetic_images

    pattern = "y{y:d+}_r{r:d+}_c0.tif"

    result = runner.invoke(
        app,
        [
            "--trainingImages",
            training_images,
            "--trainingLabels",
            training_labels,
            "--testingImages",
            test_images,
            "--testingLabels",
            test_labels,
            "--modelBackbone",
            model_backbone,
            "--filePattern",
            pattern,
            "--iterations",
            1,
            "--batchSize",
            batch_size,
            "--outDir",
            output_directory,
        ],
    )
    pathlib.Path(pathlib.Path.cwd(), "model_weights.h5").unlink()

    assert result.exit_code == 0
    clean_directories()
