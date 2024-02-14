"""Pixel segmentation eval package."""

import os
import pathlib
import shutil
import tempfile
from collections.abc import Generator

import pytest
import skimage
import vaex
from polus.plugins.features.pixel_segmentation_eval.__main__ import app
from polus.plugins.features.pixel_segmentation_eval.evaluate import evaluation
from skimage import io
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture()
def gt_dir() -> Generator[str, None, None]:
    """Create directory for saving groundtruth images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def pred_dir() -> Generator[str, None, None]:
    """Create directory for saving predicted images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Create output directory."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture(params=[512, 256, 1024])
def image_sizes(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def synthetic_images(
    gt_dir: pathlib.Path,
    pred_dir: pathlib.Path,
    image_sizes: pytest.FixtureRequest,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate random synthetic images."""
    for i in range(3):
        blobs = skimage.data.binary_blobs(
            length=image_sizes,
            volume_fraction=0.02,
            blob_size_fraction=0.02,
        )
        lab_blobs = skimage.measure.label(blobs)
        outname = f"blob_image_{i}.tif"
        gtname = pathlib.Path(gt_dir, outname)
        predname = pathlib.Path(pred_dir, outname)
        io.imsave(gtname, lab_blobs)
        io.imsave(predname, lab_blobs)

    return gt_dir, pred_dir


@pytest.fixture(
    params=[
        (".+", 1, True, False),
        (".+", 1, True, True),
        (".+", 1, False, False),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


def test_evaluation(
    synthetic_images: tuple[pathlib.Path, pathlib.Path],
    output_directory: pytest.FixtureRequest,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test Evalulation.

    This unit test runs the evaluation and validates the outputs file formats
    """
    (
        pattern,
        classes,
        indstat,
        totalstats,
    ) = get_params
    gt_dir, pred_dir = synthetic_images

    evaluation(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        input_classes=classes,
        file_pattern=pattern,
        individual_stats=indstat,
        total_stats=totalstats,
        out_dir=output_directory,
    )

    file = output_directory.joinpath("result.csv")
    vdf = vaex.open(file)
    assert vdf.shape[0] == len(os.listdir(pred_dir))


def test_cli(
    synthetic_images: tuple[pathlib.Path, pathlib.Path],
    output_directory: pytest.FixtureRequest,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test Cli."""
    pattern, classes, _, _ = get_params
    gt_dir, pred_dir = synthetic_images

    result = runner.invoke(
        app,
        [
            "--gtDir",
            gt_dir,
            "--predDir",
            pred_dir,
            "--inputClasses",
            classes,
            "--filePattern",
            pattern,
            "--individualStats",
            "--totalStats",
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
