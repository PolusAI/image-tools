"""Tests for the tool."""


import pathlib
import shutil
import tempfile
import typing

import bfio
import numpy
import pytest
import typer.testing
from polus.images.segmentation.imagej_threshold_apply.__main__ import app
from polus.images.segmentation.imagej_threshold_apply.__main__ import main


def gen_image(
    height: int,
    width: int,
    min_val: float,
    max_val: float,
    dtype: numpy.dtype,
) -> numpy.ndarray:
    """Generate a random image.

    Args:
        height: The height of the image.
        width: The width of the image.
        min_val: The minimum value of the image.
        max_val: The maximum value of the image.
        dtype: The data type of the image.

    Returns:
        The generated image.
    """
    rng = numpy.random.default_rng()
    img = rng.uniform(0.0, 1.0, size=(height, width))
    img = img * (max_val - min_val) + min_val
    return img.astype(dtype)


@pytest.fixture(
    params=[
        (0, 100, numpy.uint8),
        (-20, 100, numpy.int8),
        (0, 100, numpy.uint16),
        (-20, 100, numpy.int16),
        (0, 100, numpy.uint32),
        (-20, 100, numpy.int32),
        (0, 100, numpy.float32),
        (-20, 100, numpy.float64),
    ],
)
def gen_data(
    request: pytest.FixtureRequest,
) -> typing.Generator[
    tuple[pathlib.Path, float, pathlib.Path, pathlib.Path],
    None,
    None,
]:
    """Generate a random image for testing.

    Returns:
        A tuple containing:
            - The path to the input directory.
            - The threshold value.
            - The path to the output directory.
            - The path to the expected output image.
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
    img = gen_image(2048, 2048, min_val, max_val, dtype)

    with bfio.BioWriter(img_path) as writer:
        writer.dtype = img.dtype
        writer.Y = img.shape[0]
        writer.X = img.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:, :, 0, 0, 0] = img[:]

    threshold = (max_val - min_val) / 2
    if dtype != numpy.float32 and dtype != numpy.float64:
        threshold = float(int(threshold))

    out_img_path = data_dir.joinpath("expected_img.ome.tif")
    out_img = (img > threshold).astype(img.dtype)

    with bfio.BioWriter(out_img_path) as writer:
        writer.dtype = out_img.dtype
        writer.Y = out_img.shape[0]
        writer.X = out_img.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:, :, 0, 0, 0] = out_img[:]

    yield inp_dir, threshold, out_dir, out_img_path

    shutil.rmtree(data_dir)


def test_imagej_tool(
    gen_data: tuple[pathlib.Path, float, pathlib.Path, pathlib.Path],
) -> None:
    """Test the tool."""
    inp_dir, threshold, out_dir, out_img_path = gen_data

    main(inp_dir, ".*", threshold, out_dir)

    with bfio.BioReader(out_dir.joinpath("img.ome.tif")) as reader:
        out_img = reader[:]

    with bfio.BioReader(out_img_path) as reader:
        expected_img = reader[:]

    numpy.testing.assert_array_equal(out_img, expected_img)


def test_cli(gen_data: tuple[pathlib.Path, float, pathlib.Path, pathlib.Path]) -> None:
    """Test the CLI."""
    inp_dir, threshold, out_dir, out_img_path = gen_data

    args = [
        "--inpDir",
        str(inp_dir),
        "--pattern",
        ".*",
        "--threshold",
        str(threshold),
        "--outDir",
        str(out_dir),
    ]

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.stdout

    with bfio.BioReader(out_dir.joinpath("img.ome.tif")) as reader:
        out_img = reader[:]

    with bfio.BioReader(out_img_path) as reader:
        expected_img = reader[:]

    numpy.testing.assert_array_equal(out_img, expected_img)
