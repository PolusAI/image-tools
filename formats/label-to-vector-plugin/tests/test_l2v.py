"""Tests the label-to-vector plugin."""

import pathlib
import shutil

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.formats.label_to_vector import convert
from polus.plugins.formats.label_to_vector.__main__ import app
from polus.plugins.formats.label_to_vector.utils import helpers
from skimage import data as sk_data
from skimage.measure import label as sk_label

fixture_params = [
    (
        r"img_x{x}.ome.tif",
        list(range(4)),
        1080,
        pathlib.Path(__file__).parent.parent.joinpath("data"),
    ),
]
fixture_params.append(
    pytest.param(
        fixture_params[0],
        marks=pytest.mark.xfail(
            reason="Something went wrong with label_to_vector.",
        ),
    ),
)


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


@pytest.fixture(params=fixture_params)
def gen_images(
    request: pytest.FixtureRequest,
) -> tuple[pathlib.Path, str, pathlib.Path]:
    """Generate some images with random blobs for testing the methods in the plugin."""

    (pattern, variables, size, data_dir) = request.param

    inp_dir = data_dir.joinpath("input")
    inp_dir.mkdir(exist_ok=True)

    out_dir = data_dir.joinpath("output")
    if out_dir.exists():
        for zarr_file in out_dir.iterdir():
            shutil.rmtree(zarr_file)
    else:
        out_dir.mkdir()

    for x in variables:
        name = pattern.format(x=x)
        path = inp_dir.joinpath(name)
        if not path.exists():
            _generate_random_masks(path, size)

    return inp_dir, pattern, out_dir


def test_convert(gen_images: type[pytest.FixtureRequest]) -> None:
    """Test the label_to_vector convert method."""

    inp_dir, _, _ = gen_images

    for file in inp_dir.iterdir():
        with bfio.BioReader(file) as reader:
            labels = numpy.squeeze(reader[:])

        masks = convert(labels, file.name)
        (d, y, x) = masks.shape

        assert d == 2, "The number of vector-field dimensions is not 2."
        assert (
            y,
            x,
        ) == labels.shape, (
            "The vector-field dimensions do not match the label dimensions."
        )


def test_cli(gen_images: type[pytest.FixtureRequest]) -> None:
    """Test the label_to_vector CLI."""

    inp_dir, pattern, out_dir = gen_images
    runner = typer.testing.CliRunner()

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--filePattern",
            pattern,
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, "The CLI exited with a non-zero exit code."

    for in_path in inp_dir.iterdir():
        out_name = helpers.replace_extension(in_path, extension="_flow.ome.zarr")
        out_path = out_dir.joinpath(out_name)

        assert out_path.exists(), f"The output file does not exist for {in_path.name}."
