"""Tests for the plugin."""

import itertools
import logging
import pathlib
import shutil
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.transforms.images.apply_flatfield import apply
from polus.plugins.transforms.images.apply_flatfield.__main__ import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


FixtureReturnType = tuple[pathlib.Path, str, pathlib.Path, str]


def gen_once(num_groups: int, img_size: int) -> FixtureReturnType:
    """Generate a set of random images for testing."""

    img_pattern = "img_x{x}_c{c}.ome.tif"
    ff_pattern = "img_x(1-10)_c{c}"

    img_dir = pathlib.Path(tempfile.mkdtemp(suffix="img_dir"))
    ff_dir = pathlib.Path(tempfile.mkdtemp(suffix="ff_dir"))

    rng = numpy.random.default_rng(42)

    for i in range(num_groups):
        ff_path = ff_dir.joinpath(f"{ff_pattern.format(c=i + 1)}_flatfield.ome.tif")
        _make_random_image(ff_path, rng, img_size)

        df_path = ff_dir.joinpath(f"{ff_pattern.format(c=i + 1)}_darkfield.ome.tif")
        _make_random_image(df_path, rng, img_size)

        for j in range(10):  # 10 images in each group
            img_path = img_dir.joinpath(img_pattern.format(x=j + 1, c=i + 1))
            _make_random_image(img_path, rng, img_size)

    image_names = list(sorted(p.name for p in img_dir.iterdir()))
    logger.debug(f"Generated {image_names} images in {img_dir}")

    ff_names = list(sorted(p.name for p in ff_dir.iterdir()))
    logger.debug(f"Generated {ff_names} flatfield images in {ff_dir}")

    img_pattern = "img_x{x:d+}_c{c:d}.ome.tif"
    ff_pattern = "img_x\\(1-10\\)_c{c:d}"
    return img_dir, img_pattern, ff_dir, ff_pattern


NUM_GROUPS = [1, 4]
IMG_SIZES = [1024, 4096]
PARAMS = list(itertools.product(NUM_GROUPS, IMG_SIZES))
IDS = [f"{num_groups}_{img_size}" for num_groups, img_size in PARAMS]


@pytest.fixture(params=PARAMS, ids=IDS)
def gen_images(request: pytest.FixtureRequest) -> FixtureReturnType:
    """Generate a set of random images for testing."""
    num_groups: int
    img_size: int
    num_groups, img_size = request.param
    img_dir, img_pattern, ff_dir, ff_pattern = gen_once(num_groups, img_size)

    yield img_dir, img_pattern, ff_dir, ff_pattern

    # Cleanup
    shutil.rmtree(img_dir)
    shutil.rmtree(ff_dir)


def test_estimate(gen_images: FixtureReturnType) -> None:
    """Test the `estimate` function."""

    img_dir, img_pattern, ff_dir, ff_pattern = gen_images
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="out_dir"))

    apply(
        img_dir=img_dir,
        img_pattern=img_pattern,
        ff_dir=ff_dir,
        ff_pattern=f"{ff_pattern}_flatfield.ome.tif",
        df_pattern=f"{ff_pattern}_darkfield.ome.tif",
        out_dir=out_dir,
    )

    img_names = [p.name for p in img_dir.iterdir()]
    out_names = [p.name for p in out_dir.iterdir()]

    for name in img_names:
        assert name in out_names, f"{name} not in {out_names}"

    shutil.rmtree(out_dir)


def test_cli() -> None:
    """Test the CLI."""

    img_dir, img_pattern, ff_dir, ff_pattern = gen_once(2, 2_048)
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="out_dir"))

    runner = typer.testing.CliRunner()

    result = runner.invoke(
        app,
        [
            "--imgDir",
            str(img_dir),
            "--imgPattern",
            img_pattern,
            "--ffDir",
            str(ff_dir),
            "--ffPattern",
            f"{ff_pattern}_flatfield.ome.tif",
            "--dfPattern",
            f"{ff_pattern}_darkfield.ome.tif",
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    img_paths = set(p.name for p in img_dir.iterdir() if p.name.endswith(".ome.tif"))

    out_names = set(p.name for p in out_dir.iterdir() if p.name.endswith(".ome.tif"))

    assert img_paths == out_names, f"{(img_paths)} != {out_names}"

    # Cleanup
    shutil.rmtree(img_dir)
    shutil.rmtree(ff_dir)
    shutil.rmtree(out_dir)
