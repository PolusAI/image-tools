"""Tests for the plugin.

Since we rely on `basicpy` for the calculation of flatfield and darkfield
components, we only need to test that the plugin is able to run the
`estimate` function and that the CLI is able to run the plugin, both of which
should produce and save the two components as output images.
"""

import pathlib
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.regression.basic_flatfield_estimation import estimate
from polus.plugins.regression.basic_flatfield_estimation import utils
from polus.plugins.regression.basic_flatfield_estimation.__main__ import app

fixture_params = [
    (
        "img_c{c:d}.ome.tif",
        list(range(10)),
        1080,
    ),
]


def _make_random_image(
    path: pathlib.Path,
    rng: numpy.random.Generator,
    size: int,
) -> None:
    with bfio.BioWriter(path) as writer:
        writer.X = size
        writer.Y = size
        writer.dtype = numpy.float32

        writer[:] = rng.random(size=(size, size), dtype=writer.dtype)


@pytest.fixture(params=fixture_params)
def gen_images(
    request: pytest.FixtureRequest,
) -> tuple[str, pathlib.Path]:
    """Generate a set of random images for testing."""
    pattern: str
    variables: list[int]
    size: int
    pattern, variables, size = request.param

    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="inp_dir"))

    rng = numpy.random.default_rng(42)

    # Generate a list of file names
    files = [pattern.format(c=v) for v in variables]
    for file in files:
        path = inp_dir.joinpath(file)
        _make_random_image(path, rng, size)

    return pattern, inp_dir


def test_estimate(gen_images: tuple[str, pathlib.Path]) -> None:
    """Test the `estimate` function."""

    _, inp_dir = gen_images  # type: ignore[misc]
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="out_dir"))

    paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))
    estimate(paths, out_dir, get_darkfield=True, extension=".ome.tif")

    base_output = utils.get_output_path(paths)
    suffix = utils.get_suffix(base_output)
    flatfield_out = base_output.replace(suffix, "_flatfield.ome.tif")
    darkfield_out = base_output.replace(suffix, "_darkfield.ome.tif")

    out_names = [p.name for p in out_dir.iterdir()]
    assert (
        flatfield_out in out_names
    ), f"{flatfield_out} not in {out_names}"  # noqa: S101
    assert (
        darkfield_out in out_names
    ), f"{darkfield_out} not in {out_names}"  # noqa: S101


def test_cli(gen_images: tuple[str, pathlib.Path]) -> None:
    """Test the CLI."""
    pattern, inp_dir = gen_images  # type: ignore[misc]
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="out_dir"))

    runner = typer.testing.CliRunner()

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--outDir",
            str(out_dir),
            "--filePattern",
            pattern,
            "--groupBy",
            "",
            "--getDarkfield",
        ],
    )

    assert result.exit_code == 0  # noqa: S101

    paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))
    base_output = utils.get_output_path(paths)
    suffix = utils.get_suffix(base_output)
    flatfield_out = base_output.replace(suffix, "_flatfield.ome.tif")
    darkfield_out = base_output.replace(suffix, "_darkfield.ome.tif")

    out_names = [p.name for p in out_dir.iterdir()]
    assert (
        flatfield_out in out_names
    ), f"{flatfield_out} not in {out_names}"  # noqa: S101
    assert (
        darkfield_out in out_names
    ), f"{darkfield_out} not in {out_names}"  # noqa: S101
