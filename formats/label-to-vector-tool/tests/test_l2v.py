"""Tests the label-to-vector plugin."""

import pathlib
import shutil
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.images.formats.label_to_vector import convert
from polus.images.formats.label_to_vector.__main__ import app
from polus.images.formats.label_to_vector.utils import helpers
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
) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate some images with random blobs for testing the methods in the plugin."""

    (name, size) = request.param
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("input")
    inp_dir.mkdir()

    out_dir = data_dir.joinpath("output")
    out_dir.mkdir()

    _generate_random_masks(inp_dir.joinpath(name), size)

    yield inp_dir, out_dir

    shutil.rmtree(data_dir)


def check_single(labels: numpy.ndarray, masks: numpy.ndarray) -> bool:
    """Test the label_to_vector single method."""

    (d, y, x) = masks.shape

    assert d == 2, "The number of vector-field dimensions is not 2."
    assert (
        y,
        x,
    ) == labels.shape, "The vector-field dimensions do not match the label dimensions."
    assert masks.dtype == numpy.float32, "The vector-field dtype is not float32."

    background = labels == 0
    assert numpy.allclose(
        masks[0][background], 0
    ), "The background x-component is not 0."
    assert numpy.allclose(
        masks[1][background], 0
    ), "The background y-component is not 0."

    return True


def test_convert(gen_images: tuple[pathlib.Path, pathlib.Path]) -> None:
    """Test the label_to_vector convert method."""

    inp_dir, _ = gen_images

    for file in inp_dir.iterdir():
        if not file.name.endswith(".ome.tif"):
            continue

        with bfio.BioReader(file) as reader:
            labels = numpy.squeeze(reader[:])

        masks = convert(labels, file.name)

        assert check_single(labels, masks), "The label_to_vector single method failed."


def test_cli(gen_images: tuple[pathlib.Path, pathlib.Path]) -> None:
    """Test the label_to_vector CLI."""

    inp_dir, out_dir = gen_images
    runner = typer.testing.CliRunner()

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--filePattern",
            PATTERN,
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, "The CLI exited with a non-zero exit code."

    for in_path in inp_dir.iterdir():
        out_name = helpers.replace_extension(in_path, extension="_flow.ome.zarr")
        out_path = out_dir.joinpath(out_name)

        assert out_path.exists(), f"The output file does not exist for {in_path.name}."

        with bfio.BioReader(in_path) as reader:
            labels = numpy.squeeze(reader[:])

        with bfio.BioReader(out_path) as reader:
            masks = numpy.squeeze(reader[:])
            masks = numpy.moveaxis(masks, -1, 0)
            masks = masks[1:3, :, :]

        assert check_single(labels, masks), "The label_to_vector single method failed."
