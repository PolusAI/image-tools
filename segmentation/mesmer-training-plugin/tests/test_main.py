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
            "efficientnetv2bs",
            2,
        ),
        (
            "efficientnetv2bm",
            5,
        ),
        (
            "efficientnetv2bl",
            2,
        ),
        (
            "efficientnetv2b3",
            5,
        ),
        (
            "efficientnetv2b2",
            2,
        ),
        (
            "efficientnetv2b1",
            5,
        ),
        (
            "efficientnetv2b0",
            2,
        ),
        (
            "efficientnetb7",
            5,
        ),
        (
            "efficientnetb6",
            2,
        ),
        (
            "efficientnetb5",
            5,
        ),
        (
            "efficientnetb4",
            2,
        ),
        (
            "efficientnetb3",
            5,
        ),
        (
            "efficientnetb2",
            2,
        ),
        (
            "efficientnetb1",
            5,
        ),
        (
            "efficientnetb0",
            2,
        ),
        (
            "mobilenet_v2",
            5,
        ),
        (
            "mobilenet_v2",
            2,
        ),
        (
            "mobilenetv2",
            5,
        ),
        (
            "mobilenet",
            2,
        ),
        (
            "nasnet_mobile",
            5,
        ),
        (
            "nasnet_large",
            2,
        ),
        (
            "vgg19",
            5,
        ),
        (
            "vgg16",
            2,
        ),
        (
            "resnet152v2",
            5,
        ),
        (
            "resnet101v2",
            2,
        ),
        (
            "resnet50v2",
            5,
        ),
        (
            "resnet152",
            2,
        ),
        (
            "resnet101",
            5,
        ),
        (
            "resnet50",
            2,
        ),
        (
            "densenet201",
            5,
        ),
        (
            "densenet169",
            2,
        ),
        (
            "densenet121",
            5,
        ),
        (
            "featurenet_3d",
            2,
        ),
        (
            "featurenet3d",
            5,
        ),
        (
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
    model_backbone, batch_size = get_params
    training_images, training_labels, test_images, test_labels = synthetic_images
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
            4,
            "--batchSize",
            batch_size,
            "--outDir",
            output_directory,
        ],
    )
    pathlib.Path(pathlib.Path.cwd(), "model_weights.h5").unlink()

    assert result.exit_code == 0
