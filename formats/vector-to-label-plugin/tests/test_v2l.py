"""Tests the label-to-vector plugin."""

import pathlib
import shutil
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.formats.label_to_vector.__main__ import main as l2v_main
from polus.plugins.formats.vector_to_label.__main__ import app
from polus.plugins.formats.vector_to_label import helpers
from polus.plugins.formats.vector_to_label.dynamics import convert
from polus.plugins.formats.label_to_vector.dynamics.label_to_vector import (
    convert as l2v_convert,
)
from skimage import data as sk_data
from skimage.measure import label as sk_label


def _generate_random_masks(out_path: pathlib.Path, size: int) -> None:
    image: numpy.ndarray = sk_label(
        label_image=sk_data.binary_blobs(
            length=size,
            blob_size_fraction=0.025,
            volume_fraction=0.25,
            seed=42,
        ),
    ).astype(numpy.uint16)
    with bfio.BioWriter(out_path) as writer:
        (y, x) = image.shape
        writer.Y = y
        writer.X = x
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]


PATTERN = "img.ome.tif"
PARAMS = [(PATTERN, 1024 * (2**s)) for s in range(3)]
IDS = [str(s) for _, s in PARAMS]


@pytest.fixture(params=PARAMS, ids=IDS)
def gen_images(
    request: pytest.FixtureRequest,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Generate some images with random blobs for testing the methods in the plugin."""

    name, size = request.param
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("input")
    inp_dir.mkdir()

    out_dir = data_dir.joinpath("output")
    out_dir.mkdir()

    _generate_random_masks(inp_dir.joinpath(name), size)

    mid_dir = data_dir.joinpath("intermediate")
    mid_dir.mkdir()
    l2v_main(inp_dir, name, mid_dir)

    assert mid_dir.joinpath("img_flow.ome.zarr").exists()

    yield inp_dir, mid_dir, out_dir

    shutil.rmtree(data_dir)


def check_labels(
    inp_labels: numpy.ndarray,
    out_labels: numpy.ndarray,
) -> True:
    """Checks the validity of reconstructed labels."""

    # Get the foreground from both labels
    inp_fore = (inp_labels > 0).astype(numpy.int8)
    out_fore = (out_labels > 0).astype(numpy.int8)

    # check that the loss is less than 1%
    diff = numpy.sum(numpy.abs(inp_fore - out_fore)) / inp_fore.size
    assert diff < 0.01, f"Loss is too large: {diff}"

    # TODO: How to check that adjacent labels are merged?
    # # check that the number of labels is the same
    # inp_num = numpy.max(inp_labels)
    # out_num = numpy.max(out_labels)

    # assert inp_num == out_num, f"Number of labels is different: {inp_num} vs {out_num}"

    return True


def test_convert(gen_images: tuple[pathlib.Path, pathlib.Path, pathlib.Path]) -> None:
    """Test the conversion method."""

    inp_dir, _, _ = gen_images

    for inp_path in inp_dir.iterdir():
        with bfio.BioReader(inp_path) as reader:
            inp_labels = numpy.squeeze(reader[:])

        assert inp_labels.ndim == 2, f"The input labels are not 2D. {inp_labels.shape}"

        mask = (inp_labels > 0)[numpy.newaxis, :, :]
        assert mask.ndim == 3, f"The mask is not 2D. {mask.shape}"

        vectors = l2v_convert(inp_labels)
        assert vectors.ndim == 3, f"The vectors are not 3D. {vectors.shape}"
        assert (
            vectors.shape[0] == 2
        ), f"The vectors do not have 2 dimensions. {vectors.shape}"

        out_labels = convert(vectors, mask, 1)

        assert check_labels(inp_labels, out_labels), "The labels are not the same."


def test_cli(gen_images: tuple[pathlib.Path, pathlib.Path, pathlib.Path]) -> None:
    """Test the label_to_vector CLI."""

    inp_dir, mid_dir, out_dir = gen_images
    runner = typer.testing.CliRunner()

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(mid_dir),
            "--filePattern",
            ".*",
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, "The CLI exited with a non-zero exit code."

    for inp_path in inp_dir.iterdir():
        out_path = out_dir.joinpath(inp_path.name)
        assert (
            out_path.exists()
        ), f"The output file does not exist for {inp_path.name}. Found {list(out_dir.iterdir())}"

        with bfio.BioReader(inp_path) as reader:
            inp_labels = reader[:]

        with bfio.BioReader(out_path) as reader:
            out_labels = reader[:]

        assert check_labels(inp_labels, out_labels), "The labels are not the same."
