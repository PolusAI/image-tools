"""Tests for the plugin.

We rely on `theia` to have tested for the correctness of the neural network and
image generation. We only need to test that the plugin correctly calls `theia`
and that the output files are produced as expected.
"""

import itertools
import pathlib
import shutil
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.regression.theia_bleedthrough_estimation import model as theia
from polus.plugins.regression.theia_bleedthrough_estimation import tile_selectors
from polus.plugins.regression.theia_bleedthrough_estimation.__main__ import app
from skimage import data as sk_data

PATTERN = "blobs_c{c:d}.ome.tif"


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

    image: numpy.ndarray = sk_data.binary_blobs(
        length=length,
        blob_size_fraction=0.025,
        volume_fraction=0.25,
        seed=42,
    ).astype(numpy.float32)

    image = image / image.max()

    noise = numpy.random.poisson(image)
    noise = ((noise / noise.max()) / c_max).astype(numpy.float32)

    image = numpy.clip((image / c_max) + noise, 0.0, 1.0)

    inp_path = inp_dir.joinpath(pattern.format(c=c))
    with bfio.BioWriter(inp_path) as writer:
        (y, x) = image.shape
        writer.Y = y
        writer.X = x
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]

    assert inp_path.exists(), f"Could not create {inp_path}."
    print(f"Created {inp_path}.")


def gen_once(
    pattern: str = PATTERN,
    num_fluorophores: int = 4,
    length: int = 1_024,
    selector: tile_selectors.Selectors = tile_selectors.Selectors.MeanIntensity,
) -> tuple[pathlib.Path, pathlib.Path, str, int]:
    """Generate images for testing."""

    data_dir = pathlib.Path(__file__).parent.parent.joinpath("data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find {data_dir}.")

    inp_dir = data_dir.joinpath("input")
    if inp_dir.exists():
        shutil.rmtree(inp_dir)
        inp_dir.mkdir()

    out_dir = data_dir.joinpath("output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir()

    # inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))
    for c in range(1, num_fluorophores + 1):
        _make_blobs(inp_dir, length, pattern, c, num_fluorophores)

    # out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    return inp_dir, out_dir, pattern, num_fluorophores, selector


MAX_FLUOROPHORES = 5
NUM_FLUOROPHORES = list(range(2, MAX_FLUOROPHORES + 1))
IMG_SIZES = [1_024 * i for i in range(1, 4)]

SELECTORS = tile_selectors.Selectors.variants()
OVERLAP = list(range(1, MAX_FLUOROPHORES))
KERNEL_SIZE = [3, 5, 7]
INTERACTIONS = [True, False]

PARAMS = [
    (n, l, s, o, k, i)
    for n, l, s, o, k, i in itertools.product(
        NUM_FLUOROPHORES,
        IMG_SIZES,
        SELECTORS,
        OVERLAP,
        KERNEL_SIZE,
        INTERACTIONS,
    )
    if o < n and o % 2 == 1
]
IDS = [f"{n}_{l}_{s.value.lower()}_{o}_{k}_{int(i)}" for n, l, s, o, k, i in PARAMS]


@pytest.fixture(
    params=[(PATTERN, *p) for p in PARAMS],
    ids=IDS,
)
def gen_images(request: pytest.FixtureRequest) -> tuple[pathlib.Path, str, int]:
    """Generate images for testing."""

    inp_dir, out_dir, pattern, num_fluorophores, selector = gen_once(*request.param)

    yield inp_dir, out_dir, pattern, num_fluorophores, selector

    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


# @pytest.mark.skip(reason="Testing only the CLI first.")
# def test_lumos(gen_images: tuple[pathlib.Path, str, int]) -> None:
#     """Test that `theia` produces the correct output files."""

#     inp_dir, out_dir, _, _, selector = gen_images

#     inp_paths = list(filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir()))

#     theia.estimate_bleedthrough(
#         image_paths=inp_paths,
#         channel_order=None,
#         selection_criterion=selector,
#         channel_overlap=1,
#         kernel_size=3,
#         remove_interactions=False,
#         out_dir=out_dir,
#     )

#     for inp_path in inp_paths:
#         out_path = out_dir.joinpath(inp_path.name)

#         assert out_path.exists(), f"output of {out_path.name} does not exist."


def test_cli() -> None:
    """Test the CLI for the plugin."""

    inp_dir, out_dir, pattern, _, selector = gen_once()

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
            "c",
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
        assert result.exit_code == 0, result.stdout

        inp_paths = list(
            filter(lambda p: p.name.endswith(".ome.tif"), inp_dir.iterdir())
        )

        for inp_path in inp_paths:
            out_path = out_dir.joinpath(inp_path.name)

            assert out_path.exists(), f"output of {out_path.name} does not exist."

    except AssertionError:
        raise

    # finally:
    #     shutil.rmtree(inp_dir)
    #     shutil.rmtree(out_dir)


"""
python -m src.polus.plugins.regression.theia_bleedthrough_estimation \
    --inpDir ./data/input \
    --filePattern "S1_R{r:d}_C1-C11_A1_y009_x009_c{c:ddd}.ome.tif" \
    --groupBy "r" \
    --channelOrdering "1,0,3,2,4,5,7,6,8,9" \
    --selectionCriterion MeanIntensity \
    --channelOverlap 1 \
    --kernelSize 3 \
    --outDir ./data/output
"""
