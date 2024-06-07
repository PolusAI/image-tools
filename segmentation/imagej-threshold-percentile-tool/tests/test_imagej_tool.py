"""Tests for the tool."""


import pathlib
import shutil
import tempfile
import typing

import bfio
import numpy
import pytest
import typer.testing
from polus.images.segmentation.imagej_threshold_percentile.__main__ import app
from polus.images.segmentation.imagej_threshold_percentile.__main__ import main
from skimage.data import binary_blobs


def gen_image(
    height: int,
    width: int,
    min_val: float,
    max_val: float,
    dtype: numpy.dtype,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Generate a random image.

    Args:
        height: The height of the image.
        width: The width of the image.
        min_val: The minimum value of the image.
        max_val: The maximum value of the image.
        dtype: The data type of the image.

    Returns:
        The generated image and the mask.
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
    return img.astype(dtype), mask.astype(dtype)


@pytest.fixture(
    params=[
        (0, 100, numpy.uint8),
        (-20, 100, numpy.int8),
        (0, 100, numpy.uint16),
        (-20, 100, numpy.int16),
        (0, 100, numpy.uint32),
        (-20, 100, numpy.int32),
        (-20, 100, numpy.float64),
    ],
)
def gen_data(
    request: pytest.FixtureRequest,
) -> typing.Generator[tuple[pathlib.Path, numpy.ndarray, pathlib.Path], None, None]:
    """Generate a random image for testing.

    Returns:
        A tuple containing:
            - The path to the input directory.
            - The expected mask.
            - The path to the output directory.
    """
    min_val: float
    max_val: float
    dtype: numpy.dtype
    min_val, max_val, dtype = request.param

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("input")
    inp_dir.mkdir()

    out_dir = data_dir.joinpath("output")
    out_dir.mkdir()

    img_path = inp_dir.joinpath("img.ome.tif")
    img, mask = gen_image(2048, 2048, min_val, max_val, dtype)

    with bfio.BioWriter(img_path) as writer:
        writer.dtype = img.dtype
        writer.Y = img.shape[0]
        writer.X = img.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:, :, 0, 0, 0] = img[:]

    yield inp_dir, mask, out_dir

    shutil.rmtree(data_dir)


def test_imagej_tool(
    gen_data: tuple[pathlib.Path, numpy.ndarray, pathlib.Path],
) -> None:
    """Test the tool."""
    inp_dir, mask, out_dir = gen_data

    main(inp_dir, ".*", out_dir)

    with bfio.BioReader(out_dir.joinpath("img.ome.tif")) as reader:
        out_img = reader[:]

    # Check that the output image is similar to the mask (less than 26% difference)
    diff = (out_img != mask).mean()
    assert diff < 0.26, f"diff: {diff}"


def test_cli(gen_data: tuple[pathlib.Path, numpy.ndarray, pathlib.Path]) -> None:
    """Test the CLI."""
    inp_dir, mask, out_dir = gen_data

    args = [
        "--inpDir",
        str(inp_dir),
        "--pattern",
        ".*",
        "--outDir",
        str(out_dir),
    ]

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.stdout

    with bfio.BioReader(out_dir.joinpath("img.ome.tif")) as reader:
        out_img = reader[:]

    diff = (out_img != mask).mean()
    assert diff < 0.26, f"diff: {diff}"
