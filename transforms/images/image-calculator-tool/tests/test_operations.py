"""Tests for the image calculator plugin."""

import itertools
import pathlib
import random
import shutil
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.images.transforms.images import image_calculator
from polus.images.transforms.images.image_calculator.__main__ import app


def _make_random_image(
    path: pathlib.Path,
    rng: numpy.random.Generator,
    size: int,
) -> None:
    with bfio.BioWriter(path) as writer:
        writer.X = size
        writer.Y = size
        writer.dtype = numpy.uint32

        writer[:] = rng.integers(2**8, 2**16, size=(size, size), dtype=writer.dtype)


def gen_images(
    size: int,
) -> tuple[
    str,  # pattern
    pathlib.Path,  # primary_dir
    pathlib.Path,  # secondary_dir
    pathlib.Path,  # out_dir
]:
    """Generate a set of random images for testing."""

    pattern = "img_c{c}.ome.tif"

    num_images = 3

    # make a temporary directory
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    primary_dir = data_dir.joinpath("primary")
    primary_dir.mkdir(exist_ok=True)

    secondary_dir = data_dir.joinpath("secondary")
    secondary_dir.mkdir(exist_ok=True)

    rng = numpy.random.default_rng(42)

    # Generate a list of file names
    names = [pattern.format(c=v + 1) for v in range(num_images)]
    for name in names:
        _make_random_image(primary_dir.joinpath(name), rng, size)
        _make_random_image(secondary_dir.joinpath(name), rng, size)

    out_dir = data_dir.joinpath("outputs")
    out_dir.mkdir(exist_ok=True)

    pattern = "img_c{c:d}.ome.tif"

    return pattern, primary_dir, secondary_dir, out_dir


def _test_cli(
    pattern: str,
    primary_dir: pathlib.Path,
    secondary_dir: pathlib.Path,
    out_dir: pathlib.Path,
    op: image_calculator.Operation,
) -> None:
    """Test the CLI."""

    args = [
        "--primaryDir",
        str(primary_dir),
        "--primaryPattern",
        pattern,
        "--operator",
        op.value,
        "--secondaryDir",
        str(secondary_dir),
        "--secondaryPattern",
        pattern,
        "--outDir",
        str(out_dir),
    ]

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, args)

    assert result.exit_code == 0, f"CLI failed with {result.stdout}\n{args}"

    args = "\n".join(args)
    p_files = [p.name for p in primary_dir.iterdir() if p.name.endswith(".ome.tif")]
    o_files = [p.name for p in out_dir.iterdir() if p.name.endswith(".ome.tif")]

    for p in p_files:
        assert p in o_files, f"Missing {p} from {p_files} in {o_files}\n{args}"

    shutil.rmtree(primary_dir.parent)


@pytest.mark.parametrize("op", image_calculator.Operation.variants())
def test_cli_small(op: image_calculator.Operation) -> None:
    pattern, primary_dir, secondary_dir, out_dir = gen_images(1024)
    _test_cli(pattern, primary_dir, secondary_dir, out_dir, op)


@pytest.mark.skipif("not config.getoption('slow')")
@pytest.mark.parametrize("op", image_calculator.Operation.variants())
def test_cli_large(op: image_calculator.Operation) -> None:
    pattern, primary_dir, secondary_dir, out_dir = gen_images(1024 * 16)
    _test_cli(pattern, primary_dir, secondary_dir, out_dir, op)
