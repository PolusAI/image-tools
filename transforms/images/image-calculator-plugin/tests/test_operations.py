"""Tests for the image calculator plugin."""

import pathlib
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.transforms.images import image_calculator
from polus.plugins.transforms.images.image_calculator.__main__ import app

fixture_params = [
    (
        r"img_c{c}.ome.tif",
        list(range(4)),
        1080,
    ),
]
fixture_params.append(
    pytest.param(
        fixture_params[0],
        marks=pytest.mark.xfail(
            reason="Something went wrong with the image calculator.",
        ),
    ),
)


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


@pytest.fixture(params=fixture_params)
def gen_images(
    request: pytest.FixtureRequest,
) -> tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]:
    """Generate a set of random images for testing."""
    pattern: str
    variables: list[int]
    size: int
    pattern, variables, size = request.param
    data_dir = pathlib.Path(tempfile.mkdtemp())

    primary_dir = data_dir.joinpath("primary")
    primary_dir.mkdir(exist_ok=True)

    secondary_dir = data_dir.joinpath("secondary")
    secondary_dir.mkdir(exist_ok=True)

    rng = numpy.random.default_rng(42)

    # Generate a list of file names
    files = [pattern.format(c=v) for v in variables]
    for file in files:
        path = primary_dir.joinpath(file)
        if not path.exists():
            _make_random_image(path, rng, size)

        path = secondary_dir.joinpath(file)
        if not path.exists():
            _make_random_image(path, rng, size)

    out_dir = data_dir.joinpath("outputs")
    if out_dir.exists():
        for file in out_dir.iterdir():  # type: ignore[assignment]
            file.unlink()  # type: ignore[attr-defined]
    else:
        out_dir.mkdir()

    return pattern, primary_dir, secondary_dir, out_dir


def test_cli(gen_images: type[pytest.FixtureRequest]) -> None:
    """Test the CLI."""
    pattern: str
    primary_dir: pathlib.Path
    secondary_dir: pathlib.Path
    out_dir: pathlib.Path
    pattern, primary_dir, secondary_dir, out_dir = gen_images  # type: ignore[misc]

    runner = typer.testing.CliRunner()

    for operation in image_calculator.Operation:
        result = runner.invoke(
            app,
            [
                "--primaryDir",
                str(primary_dir),
                "--primaryPattern",
                pattern,
                "--operator",
                operation.value,
                "--secondaryDir",
                str(secondary_dir),
                "--secondaryPattern",
                pattern,
                "--outDir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0  # noqa: S101

        for path in primary_dir.iterdir():
            if path.name.endswith(".ome.tif"):
                assert out_dir.joinpath(path.name).exists()  # noqa: S101
