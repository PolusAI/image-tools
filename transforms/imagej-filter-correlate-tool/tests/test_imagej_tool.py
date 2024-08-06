"""Tests for the tool."""


import pathlib
import shutil
import tempfile
import typing

import bfio
import numpy
import pytest
import typer.testing
from polus.images.transforms.imagej_filter_correlate.__main__ import app
from polus.images.transforms.imagej_filter_correlate.__main__ import main
from skimage.data import binary_blobs


def gen_image(
    height: int,
    width: int,
    min_val: float,
    max_val: float,
    dtype: numpy.dtype,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Generate a random image and a filter.

    Args:
        height: The height of the image.
        width: The width of the image.
        min_val: The minimum value of the image.
        max_val: The maximum value of the image.
        dtype: The data type of the image.

    Returns:
        The generated image and filter.
    """
    rng = numpy.random.default_rng()
    background = rng.poisson(2, (height, width)).astype(numpy.float32)
    foreground = rng.poisson(10, (height, width)).astype(numpy.float32)
    mask: numpy.ndarray = binary_blobs(
        length=height,
        blob_size_fraction=32 / height,
        n_dim=2,
        volume_fraction=0.2,
    ).astype(numpy.float32)
    img: numpy.ndarray = (background + foreground * mask).astype(numpy.float32)
    img = (img - numpy.min(img)) / (numpy.max(img) - numpy.min(img))
    img = img * (max_val - min_val) + min_val

    filter_img = numpy.ones((3, 3), dtype=numpy.float32)
    filter_img /= filter_img.sum()

    return img.astype(dtype), filter_img.astype(dtype)


@pytest.fixture(
    params=[
        (0, 100, numpy.float32),
        (0, 100, numpy.float64),
    ],
)
def gen_data(
    request: pytest.FixtureRequest,
) -> typing.Generator[tuple[pathlib.Path, pathlib.Path, pathlib.Path], None, None]:
    """Generate a random image for testing.

    Returns:
        A tuple containing:
            - The path to the input directory.
            - The path to the output directory.
    """
    min_val: float
    max_val: float
    dtype: numpy.dtype
    min_val, max_val, dtype = request.param

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    img, filter_img = gen_image(2048, 2048, min_val, max_val, dtype)

    inp_dir = data_dir.joinpath("input")
    inp_dir.mkdir()
    img_path = inp_dir.joinpath("img.ome.tif")
    with bfio.BioWriter(img_path) as writer:
        writer.dtype = img.dtype
        writer.Y = img.shape[0]
        writer.X = img.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:, :, 0, 0, 0] = img[:]

    filter_dir = data_dir.joinpath("filter")
    filter_dir.mkdir()
    filter_path = filter_dir.joinpath("filter.ome.tif")
    with bfio.BioWriter(filter_path) as writer:
        writer.dtype = filter_img.dtype
        writer.Y = filter_img.shape[0]
        writer.X = filter_img.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:, :, 0, 0, 0] = filter_img[:]

    out_dir = data_dir.joinpath("output")
    out_dir.mkdir()

    yield inp_dir, filter_path, out_dir

    shutil.rmtree(data_dir)


def test_imagej_tool(
    gen_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Test the tool."""
    inp_dir, filter_path, out_dir = gen_data

    main(inp_dir, ".*", filter_path, out_dir)

    assert out_dir.joinpath("img.ome.tif").exists()

    with bfio.BioReader(out_dir.joinpath("img.ome.tif")) as br:
        img = br[:].squeeze()

    with bfio.BioReader(inp_dir.joinpath("img.ome.tif")) as br:
        img_orig = br[:].squeeze()

    diff = numpy.abs(img - img_orig).mean()
    diff /= numpy.percentile(numpy.abs(img_orig), 95)
    assert diff < 0.2


def test_cli(
    gen_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Test the CLI."""
    inp_dir, filter_path, out_dir = gen_data

    args = [
        "--inpDir",
        str(inp_dir),
        "--pattern",
        ".*",
        "--filterPath",
        str(filter_path),
        "--outDir",
        str(out_dir),
    ]

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.stdout

    assert out_dir.joinpath("img.ome.tif").exists()

    with bfio.BioReader(out_dir.joinpath("img.ome.tif")) as br:
        img = br[:].squeeze()

    with bfio.BioReader(inp_dir.joinpath("img.ome.tif")) as br:
        img_orig = br[:].squeeze()

    diff = numpy.abs(img - img_orig).mean()
    diff /= numpy.percentile(numpy.abs(img_orig), 95)
    assert diff < 0.2
