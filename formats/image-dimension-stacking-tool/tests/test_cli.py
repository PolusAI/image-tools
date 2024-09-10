"""Tests for the CLI."""


import pathlib
import tempfile

import bfio
import numpy
from polus.images.formats.image_dimension_stacking import utils
import pytest
import scipy.ndimage
import typer.testing

from polus.images.formats.image_dimension_stacking.__main__ import app


def gen_images(
    inp_dir: pathlib.Path,
    pattern: str,
    axis: str,
    num_images: int,
    size: int,
) -> None:
    """Generate synthetic images."""

    # Generate a image with a square in the middle
    square = numpy.zeros((size, size), dtype=numpy.float32)
    lq = size // 4
    hq = 3 * size // 4
    square[lq:hq, lq:hq] = 1

    for i in range(num_images):
        # Rotate the image by i degrees
        img = scipy.ndimage.rotate(square, i, reshape=False)

        name = pattern.format(**{axis: i})
        path = inp_dir / name

        with bfio.BioWriter(path) as writer:
            writer.X = size
            writer.Y = size
            writer.Z = 1
            writer.C = 1
            writer.T = 1
            writer.dtype = numpy.float32

            writer[:] = img


@pytest.mark.parametrize("axis", ["z", "c", "t"])
@pytest.mark.parametrize("num_images", [10])
@pytest.mark.parametrize("size", [1024, 1024 * 4])
def test_cli(
    axis: str,
    num_images: int,
    size: int,
) -> None:
    """Test the command line."""

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))
    inp_dir = data_dir / "input"
    out_dir = data_dir / "output"

    for d in [inp_dir, out_dir]:
        d.mkdir(exist_ok=True)

    pattern = f"image_{axis}" + "{" + f"{axis}" + ":03d}.ome.tif"
    gen_images(inp_dir, pattern, axis, num_images, size)

    pattern = f"image_{axis}" + "{" + f"{axis}" + ":d+}.ome.tif"

    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--filePattern",
            pattern,
            "--axis",
            axis,
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0

    # Check the output
    start = f"{0:03d}"
    end = f"{num_images - 1:03d}"
    out_path = out_dir / f"image_{axis}({start}-{end}).ome.tif"
    assert out_path.exists()

    with bfio.BioReader(out_path) as reader:
        assert reader.X == size
        assert reader.Y == size

        if axis == "z":
            assert reader.Z == num_images
            assert reader.C == 1
            assert reader.T == 1
        elif axis == "c":
            assert reader.Z == 1
            assert reader.C == num_images
            assert reader.T == 1
        elif axis == "t":
            assert reader.Z == 1
            assert reader.C == 1
            assert reader.T == num_images

        base = numpy.squeeze(reader[:, :, 0, 0, 0])
        for i in range(num_images):
            if axis == "z":
                img = reader[:, :, i, 0, 0]
            elif axis == "c":
                img = reader[:, :, 0, i, 0]
            elif axis == "t":
                img = reader[:, :, 0, 0, i]
            else:
                pytest.fail(f"Unknown axis {axis}")

            img = numpy.squeeze(img)
            assert numpy.allclose(img, scipy.ndimage.rotate(base, i, reshape=False))
