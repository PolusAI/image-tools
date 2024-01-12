"""Nyxus Plugin."""
from pathlib import Path
import shutil
import tempfile

import filepattern as fp
import numpy as np
import pytest
import vaex
from typing import Union
from skimage import filters, io, measure
from typer.testing import CliRunner

from polus.plugins.features.nyxus_plugin.__main__ import app as app
from polus.plugins.features.nyxus_plugin.nyxus_func import nyxus_func

runner = CliRunner()


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_nyxus_func(
    synthetic_images: tuple[Union[str, Path], Union[str, Path]],
    output_directory: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> None:
    """Test Nyxus Function.

    This unit test runs the nyxus function and validates the outputs
    """
    inp_dir, seg_dir = synthetic_images
    int_pattern = "y04_r{r:d}_c{c:d}.ome.tif"
    seg_pattern = "y04_r{r:d}_c0.ome.tif"
    int_images = fp.FilePattern(inp_dir, int_pattern)
    seg_images = fp.FilePattern(seg_dir, seg_pattern)
    fileext, EXT, feat = get_params
    for s_image in seg_images():
        i_image = int_images.get_matching(**{k: v for k, v in s_image[0].items()})
        for i in i_image:
            nyxus_func(
                int_file=i[1],
                seg_file=s_image[1],
                out_dir=output_directory,
                features=[feat],
                file_extension=fileext,
            )

    output_ext = [f.suffix for f in output_directory.iterdir()][0]
    assert output_ext == EXT
    vdf = vaex.open([f for f in output_directory.iterdir()][0])
    assert vdf.shape is not None
    clean_directories()


@pytest.fixture
def scaled_images(
    inp_dir: Union[str, Path],
    seg_dir: Union[str, Path],
    scaled_sizes: pytest.FixtureRequest,
) -> tuple[Union[str, Path], Union[str, Path]]:
    """Generate random synthetic images."""
    im = np.zeros((scaled_sizes, scaled_sizes))
    points = scaled_sizes * np.random.random((2, 1**2))
    im[(points[0]).astype(int), (points[1]).astype(int)] = 1
    im = filters.gaussian(im, sigma=scaled_sizes / (20.0 * 10))
    blobs = im > im.mean()
    lab_blobs = measure.label(blobs, background=0)
    intname = "y04_r1_c1.ome.tif"
    segname = "y04_r1_c0.ome.tif"
    int_name = Path(inp_dir, intname)
    seg_name = Path(seg_dir, segname)
    io.imsave(int_name, im)
    io.imsave(seg_name, lab_blobs)
    return inp_dir, seg_dir


@pytest.fixture(params=[("pandas", ".csv", "MEAN")])
def get_scaled_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    yield request.param


@pytest.mark.skipif("not config.getoption('slow')")
def test_scaled_nyxus_func(
    scaled_images: tuple[Union[str, Path], Union[str, Path]],
    output_directory: Union[str, Path],
    get_scaled_params: pytest.FixtureRequest,
) -> None:
    """Test Nyxus Function.

    This unit test runs the nyxus function and validates the outputs
    """
    inp_dir, seg_dir = scaled_images
    int_pattern = "y04_r{r:d}_c{c:d}.ome.tif"
    seg_pattern = "y04_r{r:d}_c0.ome.tif"
    int_images = fp.FilePattern(inp_dir, int_pattern)
    seg_images = fp.FilePattern(seg_dir, seg_pattern)
    fileext, EXT, feat = get_scaled_params
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
    assert output_ext == EXT
    vdf = vaex.open([f for f in output_directory.iterdir()][0])
    assert vdf.shape is not None
    clean_directories()


def test_cli(synthetic_images, output_directory, get_params) -> None:
    """Test Cli."""
    inp_dir, seg_dir = synthetic_images
    int_pattern = "y04_r{r:d}_c1.ome.tif"
    seg_pattern = "y04_r{r:d}_c0.ome.tif"
    fileext, _, feat = get_params

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
            "--singleRoi",
            False,
            "--outDir",
            output_directory,
        ],
    )
    assert output_directory.joinpath(f"y04_r1_c1{fileext}")
    clean_directories()
