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
from polus.plugins.transforms.images import image_calculator
from polus.plugins.transforms.images.image_calculator.__main__ import app


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


FixtureReturnType = tuple[
    str,  # pattern
    pathlib.Path,  # primary_dir
    pathlib.Path,  # secondary_dir
    pathlib.Path,  # out_dir
    image_calculator.Operation,  # operation
]
IMAGE_SIZE = [1024 * (2**i) for i in range(4)]
OPERATIONS = image_calculator.Operation.variants()
PARAMS = [
    (r"img_c{c}.ome.tif", s, o) for s, o in itertools.product(IMAGE_SIZE, OPERATIONS)
]
IDS = [f"{s}_{o.value.lower()}" for _, s, o in PARAMS]


@pytest.fixture(params=PARAMS, ids=IDS)
def gen_images(request: pytest.FixtureRequest) -> FixtureReturnType:
    """Generate a set of random images for testing."""
    pattern: str
    size: int
    op: image_calculator.Operation
    pattern, size, op = request.param

    # make a temporary directory
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    primary_dir = data_dir.joinpath("primary")
    primary_dir.mkdir(exist_ok=True)

    secondary_dir = data_dir.joinpath("secondary")
    secondary_dir.mkdir(exist_ok=True)

    rng = numpy.random.default_rng(42)

    # Generate a list of file names
    names = [pattern.format(c=v + 1) for v in range(6)]
    for name in names:
        _make_random_image(primary_dir.joinpath(name), rng, size)
        _make_random_image(secondary_dir.joinpath(name), rng, size)

    out_dir = data_dir.joinpath("outputs")
    out_dir.mkdir(exist_ok=True)

    yield pattern, primary_dir, secondary_dir, out_dir, op

    shutil.rmtree(data_dir)


def test_cli(gen_images: FixtureReturnType) -> None:
    """Test the CLI."""
    pattern, primary_dir, secondary_dir, out_dir, op = gen_images

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
