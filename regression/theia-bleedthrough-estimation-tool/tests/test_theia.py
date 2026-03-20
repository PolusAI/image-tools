"""Tests for the plugin.

We rely on `theia` to have tested for the correctness of the neural network and
image generation. We only need to test that the plugin correctly calls `theia`
and that the output files are produced as expected.
"""

import itertools
import pathlib
import shutil
import tempfile
import typing

import bfio
import numpy
import pytest
import skimage.data
import typer.testing
from polus.images.regression.theia_bleedthrough_estimation import model as theia
from polus.images.regression.theia_bleedthrough_estimation import tile_selectors
from polus.images.regression.theia_bleedthrough_estimation.__main__ import app

PATTERN = "blobs_c{c:d}.ome.tif"
NUM_FLUOROPHORES = [2, 4]
IMG_SIZES = [2_048]
SELECTORS = [tile_selectors.Selectors.MeanIntensity, tile_selectors.Selectors.Entropy]

PARAMS = [
    (n, l, s)
    for n, l, s in itertools.product(  # noqa: E741
        NUM_FLUOROPHORES,
        IMG_SIZES,
        SELECTORS,
    )
]
IDS = [f"{n}_{l}_{s.value.lower()}" for n, l, s in PARAMS]  # noqa: E741
PARAMS = [(PATTERN, *p) for p in PARAMS]  # type: ignore[misc]


def _make_blobs(
    inp_dir: pathlib.Path,
    length: int,
    pattern: str,
    c: int,
    c_max: int,
) -> None:
    """Make a binary image with blobs.

    Args:
        inp_dir: input directory.
        length: number of rows and columns.
        pattern: file pattern.
        c: number of channels.
        c_max: maximum number of channels.
    """
    image: numpy.ndarray = skimage.data.binary_blobs(
        length=length,
        blob_size_fraction=0.025,
        volume_fraction=0.25,
        seed=42,
    ).astype(numpy.float32)

    image = image / image.max()

    noise = numpy.random.poisson(image)  # noqa: NPY002
    noise = ((noise / noise.max()) / c_max).astype(numpy.float32)

    image = numpy.clip((image / c_max) + noise, 0.0, 1.0)

    inp_path = inp_dir.joinpath(pattern.format(c=c))
    with bfio.BioWriter(inp_path) as writer:
        writer.Y = length
        writer.X = length
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]

    assert inp_path.exists(), f"Could not create {inp_path}."


def gen_once(
    pattern: str = PATTERN,
    num_fluorophores: int = 4,
    length: int = 1_024,
    selector: tile_selectors.Selectors = tile_selectors.Selectors.MeanIntensity,
) -> tuple[pathlib.Path, pathlib.Path, str, int, tile_selectors.Selectors]:
    """Generate images for testing."""
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("input")
    inp_dir.mkdir(exist_ok=True)

    out_dir = data_dir.joinpath("output")
    out_dir.mkdir(exist_ok=True)

    for c in range(1, num_fluorophores + 1):
        _make_blobs(inp_dir, length, pattern, c, num_fluorophores)

    return inp_dir, out_dir, pattern, num_fluorophores, selector


@pytest.fixture(params=PARAMS, ids=IDS)
def gen_images(
    request: pytest.FixtureRequest,
) -> typing.Generator[
    tuple[pathlib.Path, pathlib.Path, str, int, tile_selectors.Selectors],
    None,
    None,
]:
    """Generate images for testing."""
    pattern: str
    num_fluorophores: int
    img_size: int
    selector: tile_selectors.Selectors
    pattern, num_fluorophores, img_size, selector = request.param

    inp_dir, out_dir, pattern, num_fluorophores, selector = gen_once(
        pattern,
        num_fluorophores,
        img_size,
        selector,
    )

    yield inp_dir, out_dir, pattern, num_fluorophores, selector

    shutil.rmtree(inp_dir.parent)


def test_theia(
    gen_images: tuple[
        pathlib.Path,  # input directory
        pathlib.Path,  # output directory
        str,  # file pattern
        int,  # number of fluorophores
        tile_selectors.Selectors,  # selector
    ],
) -> None:
    """Test that `theia` produces the correct output files."""
    inp_dir, out_dir, _, num_fluorophores, selector = gen_images

    inp_paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))

    theia.estimate_bleedthrough(
        image_paths=inp_paths,
        channel_order=list(range(num_fluorophores)),
        selection_criterion=selector,
        channel_overlap=1,
        kernel_size=3,
        remove_interactions=False,
        verbose=0,
        out_dir=out_dir,
    )

    out_dir = out_dir.joinpath("images")

    for inp_path in inp_paths:
        out_path = out_dir.joinpath(inp_path.name)

        assert out_path.exists(), f"output of {out_path.name} does not exist."


@pytest.mark.skipif("not config.getoption('slow')")
def test_cli() -> None:
    """Test the CLI for the plugin."""
    inp_dir: pathlib.Path
    out_dir: pathlib.Path
    pattern: str
    selector: tile_selectors.Selectors

    inp_dir, out_dir, pattern, num_fluorophores, selector = gen_once()

    assert inp_dir.exists(), f"input directory {inp_dir} does not exist."
    assert out_dir.exists(), f"output directory {out_dir} does not exist."

    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--filePattern",
            pattern,
            "--groupBy",
            "",
            "--channelOrdering",
            ",".join(map(str, range(num_fluorophores))),
            "--selectionCriterion",
            selector.value,
            "--channelOverlap",
            "1",
            "--kernelSize",
            "3",
            "--outDir",
            str(out_dir),
        ],
    )

    try:
        assert result.exit_code == 0, result.stdout + result.stderr

        inp_paths = list(
            filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()),
        )

        out_dir = out_dir.joinpath("images")
        for inp_path in inp_paths:
            out_path = out_dir.joinpath(inp_path.name)

            assert out_path.exists(), f"output of {out_path.name} does not exist."

    except AssertionError:
        raise

    finally:
        shutil.rmtree(inp_dir.parent)


"""
python -m src.polus.images.regression.theia_bleedthrough_estimation \
    --inpDir ./data/input \
    --filePattern "blobs_c{c:d}.ome.tif" \
    --groupBy "" \
    --channelOrdering "0,1,2,3" \
    --selectionCriterion MeanIntensity \
    --channelOverlap 1 \
    --kernelSize 3 \
    --outDir ./data/output
"""
