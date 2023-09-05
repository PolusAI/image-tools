"""Tests for the plugin."""

import itertools
import pathlib
import shutil
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


def gen_once(
    pattern: str = "blobs_c{c:d}.ome.tif",
    num_fluorophores: int = 4,
    length: int = 1_024,
) -> tuple[pathlib.Path, pathlib.Path, str, int]:
    """Generate images for testing."""

    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))
    for c in range(1, num_fluorophores + 1):
        _make_blobs(inp_dir, length, pattern, c, num_fluorophores)

    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    return inp_dir, out_dir, pattern, num_fluorophores


NUM_FLUOROPHORES = [2, 3, 4]
IMG_SIZES = [1_024 * i for i in range(1, 4)]
PARAMS = list(itertools.product(NUM_FLUOROPHORES, IMG_SIZES))
IDS = [f"{n}_{l}" for n, l in PARAMS]


@pytest.fixture(
    params=[("blobs_c{c:d}.ome.tif", n, l) for n, l in PARAMS],
    ids=IDS,
)
def gen_images(request: pytest.FixtureRequest) -> tuple[pathlib.Path, str, int]:
    """Generate images for testing."""

    inp_dir, out_dir, pattern, num_fluorophores = gen_once(*request.param)

    yield inp_dir, out_dir, pattern, num_fluorophores

    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


def test_lumos(gen_images: tuple[pathlib.Path, str, int]) -> None:
    """Test the `correct` function."""

    inp_dir, out_dir, _, num_fluorophores = gen_images

    paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))
    output_name = utils.get_output_name(paths, ".ome.zarr")

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


def test_cli() -> None:
    """Test the CLI for the plugin."""
    inp_dir, out_dir, pattern, num_fluorophores = gen_once()

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

    try:
        assert result.exit_code == 0, result.stdout

        paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))
        out_path = out_dir.joinpath(utils.get_output_name(paths, ".ome.zarr"))
        assert out_path.exists(), f"{out_path.name} does not exist."

        with bfio.BioReader(out_path) as reader:
            assert 0 < reader.C <= num_fluorophores + 1, (
                f"Expected between 1 and {num_fluorophores + 1} channels, "
                f"but found {reader.C} channels."
            )

    except AssertionError:
        raise

    finally:
        shutil.rmtree(inp_dir)
        shutil.rmtree(out_dir)
