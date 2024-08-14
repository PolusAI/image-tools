"""Tests for the tool."""


import pathlib
import shutil
import tempfile
import typing

import bfio
import numpy
import pytest
import typer.testing
from polus.images.transforms.imagej_deconvolve_richardsonlucy.__main__ import app
from polus.images.transforms.imagej_deconvolve_richardsonlucy.__main__ import main
from scipy.signal import convolve2d
from skimage.data import binary_blobs


def gen_image(
    height: int,
    width: int,
    min_val: float,
    max_val: float,
    dtype: numpy.dtype,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Generate a random image and a PSF.

    Args:
        height: The height of the image.
        width: The width of the image.
        min_val: The minimum value of the image.
        max_val: The maximum value of the image.
        dtype: The data type of the image.

    Returns:
        The generated image, the PSF, and the original image.
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
    img_orig: numpy.ndarray = (background + foreground * mask).astype(numpy.float32)
    img_orig = (img_orig - numpy.min(img_orig)) / (
        numpy.max(img_orig) - numpy.min(img_orig)
    )
    img_orig = img_orig * (max_val - min_val) + min_val
    img_orig = img_orig.astype(dtype)

    psf = numpy.zeros_like(img_orig)
    # apply a gaussian blur to the psf
    conv = numpy.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=numpy.float32,
    )
    conv = conv / numpy.sum(conv)
    # repeat the convolution multiple times to simulate a larger kernel
    for _ in range(16):
        psf = convolve2d(psf, conv, mode="same")

    # normalize the psf
    psf = (psf - numpy.min(psf)) / (numpy.max(psf) - numpy.min(psf))

    # Convolve the image with the PSF
    img = img_orig.copy()
    for _ in range(8):
        img = convolve2d(img, conv, mode="same")

    return img, psf, img_orig


@pytest.fixture(
    params=[
        (0, 100, numpy.float32, 16),
        (-20, 100, numpy.float64, 16),
    ],
)
def gen_data(
    request: pytest.FixtureRequest,
) -> typing.Generator[
    tuple[pathlib.Path, pathlib.Path, pathlib.Path, int, numpy.ndarray],
    None,
    None,
]:
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
    max_iterations: int
    min_val, max_val, dtype, max_iterations = request.param

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    img, psf, orig = gen_image(2048, 2048, min_val, max_val, dtype)

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

    psf_dir = data_dir.joinpath("psf")
    psf_dir.mkdir()
    psf_path = inp_dir.joinpath("psf.ome.tif")
    with bfio.BioWriter(psf_path) as writer:
        writer.dtype = psf.dtype
        writer.Y = psf.shape[0]
        writer.X = psf.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:, :, 0, 0, 0] = psf[:]

    out_dir = data_dir.joinpath("output")
    out_dir.mkdir()

    yield inp_dir, psf_path, out_dir, max_iterations, orig

    shutil.rmtree(data_dir)


def test_imagej_tool(
    gen_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path, int, numpy.ndarray],
) -> None:
    """Test the tool."""
    inp_dir, psf_path, out_dir, max_iterations, _ = gen_data

    main(inp_dir, ".*", psf_path, max_iterations, out_dir)

    assert out_dir.joinpath("img.ome.tif").exists()


def test_cli(
    gen_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path, int, numpy.ndarray],
) -> None:
    """Test the CLI."""
    inp_dir, psf_path, out_dir, max_iterations, _ = gen_data

    args = [
        "--inpDir",
        str(inp_dir),
        "--pattern",
        ".*",
        "--psfPath",
        str(psf_path),
        "--maxIterations",
        str(max_iterations),
        "--outDir",
        str(out_dir),
    ]

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.stdout

    assert out_dir.joinpath("img.ome.tif").exists()
