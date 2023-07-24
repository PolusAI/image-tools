"""Tests for the plugin."""

import pathlib
import tempfile

import bfio
import filepattern
import numpy
import pytest
import typer.testing
from polus.plugins.transforms.images.lumos_bleedthrough_correction import lumos
from polus.plugins.transforms.images.lumos_bleedthrough_correction import utils
from polus.plugins.transforms.images.lumos_bleedthrough_correction.__main__ import app
from skimage import data

fixture_params = [
    (
        "blobs_c{c:d}.ome.tif",  # input image name
        4,  # number of fluorophores
        1024,  # number of rows and columns
    ),
]


def _make_blobs(
    inp_dir: pathlib.Path, length: int, pattern: str, c: int, c_max: int
) -> None:
    """Make a binary image with blobs.

    Args:
        inp_dir: input directory.
        length: number of rows and columns.
        pattern: file pattern.
        c: number of channels.
    """

    image: numpy.ndarray = data.binary_blobs(
        length=length,
        blob_size_fraction=0.025,
        volume_fraction=0.25,
        seed=42,
    ).astype(numpy.float32)

    image = (image / image.max()) / c_max
    noise = numpy.random.poisson(image)
    image = numpy.clip(image + noise, 0.0, 1.0)

    with bfio.BioWriter(inp_dir.joinpath(pattern.format(c=c))) as writer:
        (y, x) = image.shape
        writer.Y = y
        writer.X = x
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]


@pytest.fixture(params=fixture_params)
def gen_images(request: pytest.FixtureRequest) -> tuple[pathlib.Path, str, int]:
    """Generate images for testing."""

    pattern: str
    num_fluorophores: int
    length: int
    pattern, num_fluorophores, length = request.param

    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))
    for c in range(1, num_fluorophores + 1):
        _make_blobs(inp_dir, length, pattern, c, num_fluorophores)

    return inp_dir, pattern, num_fluorophores


def test_lumos(gen_images: tuple[pathlib.Path, str, int]) -> None:
    """Test the `correct` function."""

    inp_dir, _, num_fluorophores = gen_images

    paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))
    output_name = utils.get_output_name(paths, ".ome.zarr")

    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    out_path = out_dir.joinpath(output_name)

    lumos.correct(
        image_paths=paths,
        num_fluorophores=num_fluorophores,
        output_path=out_path,
    )

    assert out_path.exists(), f"{out_path.name} does not exist."

    with bfio.BioReader(out_path) as reader:
        assert 0 < reader.C <= num_fluorophores + 1, (
            f"Expected between 1 and {num_fluorophores + 1} channels, "
            f"but found {reader.C} channels."
        )


def test_cli(gen_images: tuple[pathlib.Path, str, int]) -> None:
    """Test the CLI for the plugin."""

    inp_dir, pattern, num_fluorophores = gen_images

    paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))
    output_name = utils.get_output_name(paths, ".ome.zarr")

    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    out_path = out_dir.joinpath(output_name)

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
            "--numFluorophores",
            str(num_fluorophores),
        ],
    )

    assert result.exit_code == 0, result.stdout

    assert out_path.exists(), f"{out_path.name} does not exist."

    with bfio.BioReader(out_path) as reader:
        assert 0 < reader.C <= num_fluorophores + 1, (
            f"Expected between 1 and {num_fluorophores + 1} channels, "
            f"but found {reader.C} channels."
        )
