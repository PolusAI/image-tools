"""Mesmer Training."""
import pathlib
import shutil
import tempfile
from collections.abc import Generator
from typing import Any, Sequence

import numpy as np
import pytest
from skimage import filters, io, measure
from typer.testing import CliRunner

from polus.plugins.segmentation.mesmer_training.__main__ import app as app

runner = CliRunner()


@pytest.fixture
def training_images() -> Generator[str, None, None]:
    """Create directory for saving train images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def training_labels() -> Generator[str, None, None]:
    """Create directory for saving train labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_images() -> Generator[str, None, None]:
    """Create directory for saving test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_labels() -> Generator[str, None, None]:
    """Create directory for saving test labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Create output directory."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture
def synthetic_images(
    training_images, training_labels, test_images, test_labels
) -> Generator[Sequence[Any], None, None]:
    """Generate random synthetic train images."""
    n = 20
    size = 256
    for i in range(10):
        image = np.zeros((size, size))
        points = size * np.random.random((2, n**2))
        image[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
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
    yield training_images, training_labels, test_images, test_labels


@pytest.fixture(
    params=[
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2bs",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2bm",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2bl",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2b3",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2b2",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2b1",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetv2b0",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb7",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb6",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb5",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb4",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb3",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb2",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb1",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "efficientnetb0",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "mobilenet_v2",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "mobilenet_v2",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "mobilenetv2",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "mobilenet",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "nasnet_mobile",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "nasnet_large",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "vgg19",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "vgg16",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "resnet152v2",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "resnet101v2",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "resnet50v2",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "resnet152",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "resnet101",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "resnet50",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "densenet201",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "densenet169",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "densenet121",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "featurenet_3d",
            2,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "featurenet3d",
            5,
        ),
        (
            "y{y:d+}_r{r:d+}_c0.tif",
            "featurenet",
            2,
        ),
    ]
)
def get_params(request):
    """To get the parameter of the fixture."""
    yield request.param


def test_cli(synthetic_images, output_directory, get_params) -> None:
    """Test Cli."""
    pattern, model_backbone, batch_size = get_params
    training_images, training_labels, test_images, test_labels = synthetic_images

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
            4,
            "--batchSize",
            batch_size,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
