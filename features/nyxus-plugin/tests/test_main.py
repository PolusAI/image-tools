"""Nyxus Plugin."""
import pathlib
import shutil
import tempfile
from collections.abc import Generator
from typing import Tuple

import filepattern as fp
import numpy as np
import pytest
import vaex
from skimage import filters, io, measure
from typer.testing import CliRunner

from polus.plugins.features.nyxus_plugin.__main__ import app as app
from polus.plugins.features.nyxus_plugin.nyxus_func import nyxus_func

runner = CliRunner()


@pytest.fixture
def inp_dir() -> Generator[str, None, None]:
    """Create directory for saving intensity images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def seg_dir() -> Generator[str, None, None]:
    """Create directory for saving groundtruth labelled images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Create output directory."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture(params=[256, 512, 1024, 2048])
def image_sizes(request):
    """To get the parameter of the fixture."""
    yield request.param


@pytest.fixture
def synthetic_images(
    inp_dir, seg_dir, image_sizes
) -> Generator[Tuple[pathlib.Path, pathlib.Path], None, None]:
    """Generate random synthetic images."""
    for i in range(10):
        im = np.zeros((image_sizes, image_sizes))
        points = image_sizes * np.random.random((2, 10**2))
        im[(points[0]).astype(int), (points[1]).astype(int)] = 1
        im = filters.gaussian(im, sigma=image_sizes / (20.0 * 10))
        blobs = im > im.mean()
        lab_blobs = measure.label(blobs, background=0)
        intname = f"y04_r{i}_c1.ome.tif"
        segname = f"y04_r{i}_c0.ome.tif"
        int_name = pathlib.Path(inp_dir, intname)
        seg_name = pathlib.Path(seg_dir, segname)
        io.imsave(int_name, im)
        io.imsave(seg_name, lab_blobs)
    yield inp_dir, seg_dir


@pytest.fixture(params=[(".csv", "MEAN"), (".arrow", "MEDIAN"), (".feather", "MODE")])
def get_params(request):
    """To get the parameter of the fixture."""
    yield request.param


def test_nyxus_func(synthetic_images, output_directory, get_params) -> None:
    """Test Nyxus Function.

    This unit test runs the nyxus function and validates the outputs
    """
    inp_dir, seg_dir = synthetic_images
    int_pattern = "y04_r{r:d}_c1.ome.tif"
    seg_pattern = "y04_r{r:d}_c0.ome.tif"
    int_images = fp.FilePattern(inp_dir, int_pattern)
    seg_images = fp.FilePattern(seg_dir, seg_pattern)
    fileext, feat = get_params
    for s_image in seg_images():
        i_image = int_images.get_matching(**{k: v for k, v in s_image[0].items()})
        nyxus_func(
            int_file=i_image[0][1],
            seg_file=s_image[1],
            out_dir=output_directory,
            features=[feat],
            file_extension=fileext,
        )

    output_ext = [f.suffix for f in output_directory.iterdir()][0]
    assert output_ext == fileext
    vdf = vaex.open([f for f in output_directory.iterdir()][0])
    assert vdf.shape is not None


@pytest.fixture(params=[5000, 10000, 20000, 30000])
def scaled_sizes(request):
    """To get the parameter of the fixture."""
    yield request.param


@pytest.fixture
def scaled_images(
    inp_dir, seg_dir, scaled_sizes
) -> Generator[Tuple[pathlib.Path, pathlib.Path], None, None]:
    """Generate random synthetic images."""
    im = np.zeros((scaled_sizes, scaled_sizes))
    points = scaled_sizes * np.random.random((2, 1**2))
    im[(points[0]).astype(int), (points[1]).astype(int)] = 1
    im = filters.gaussian(im, sigma=scaled_sizes / (20.0 * 10))
    blobs = im > im.mean()
    lab_blobs = measure.label(blobs, background=0)
    intname = "y04_r1_c1.ome.tif"
    segname = "y04_r1_c0.ome.tif"
    int_name = pathlib.Path(inp_dir, intname)
    seg_name = pathlib.Path(seg_dir, segname)
    io.imsave(int_name, im)
    io.imsave(seg_name, lab_blobs)
    yield inp_dir, seg_dir


@pytest.fixture(params=[(".csv", "MEAN")])
def get_scaled_params(request):
    """To get the parameter of the fixture."""
    yield request.param


def test_scaled_nyxus_func(scaled_images, output_directory, get_scaled_params) -> None:
    """Test Nyxus Function.

    This unit test runs the nyxus function and validates the outputs
    """
    inp_dir, seg_dir = scaled_images
    int_pattern = "y04_r{r:d}_c1.ome.tif"
    seg_pattern = "y04_r{r:d}_c0.ome.tif"
    int_images = fp.FilePattern(inp_dir, int_pattern)
    seg_images = fp.FilePattern(seg_dir, seg_pattern)
    fileext, feat = get_scaled_params
    for s_image in seg_images():
        i_image = int_images.get_matching(**{k: v for k, v in s_image[0].items()})
        nyxus_func(
            int_file=i_image[0][1],
            seg_file=s_image[1],
            out_dir=output_directory,
            features=[feat],
            file_extension=fileext,
        )
    output_ext = [f.suffix for f in output_directory.iterdir()][0]
    assert output_ext == fileext
    vdf = vaex.open([f for f in output_directory.iterdir()][0])
    assert vdf.shape is not None


def test_cli(synthetic_images, output_directory, get_params) -> None:
    """Test Cli."""
    inp_dir, seg_dir = synthetic_images
    int_pattern = "y04_r{r:d}_c1.ome.tif"
    seg_pattern = "y04_r{r:d}_c0.ome.tif"
    fileext, feat = get_params

    runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--segDir",
            seg_dir,
            "--intPattern",
            int_pattern,
            "--segPattern",
            seg_pattern,
            "--features",
            feat,
            "--fileExtension",
            fileext,
            "--outDir",
            output_directory,
        ],
    )
    assert output_directory.joinpath(f"y04_r1_c1{fileext}")
