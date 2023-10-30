"""Utility functions for tests."""

import json
import os
import random
import pathlib
import shutil
import tempfile

import bfio
import numpy
import pytest
import typer.testing
from polus.plugins.transforms.images.image_assembler import assemble_images
from polus.plugins.transforms.images.image_assembler.__main__ import app


def get_temp_file(path: pathlib.Path, suffix: str) -> pathlib.Path:
    """Create path to a temp file."""
    # Create a random string
    name = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))  # noqa: S311
    name += "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=15),  # noqa: S311
    )
    name += suffix
    return path.joinpath(name)


def gen_data(
    inp_dir: pathlib.Path,
    stitch_dir: pathlib.Path,
    ground_truth_dir: pathlib.Path,
) -> None:
    """Generate test data.

    Args:
        inp_dir: Directory where the partial images will be saved
        stitch_dir: Directory where the stitching vector will be saved
        ground_truth_dir: Directory where the ground truth image will be saved
    """
    # generate the image data
    tile_size = 1024
    fov_width = 1392
    fov_height = 1040
    offset_x = 1392 - tile_size
    offset_y = 1040 - tile_size
    image_width = 2 * tile_size
    image_height = 2 * tile_size
    image_shape = (image_width, image_height, 1, 1, 1)
    data = numpy.zeros(image_shape, dtype=numpy.uint8)

    # max value for np.uint8 so we have a white square in the middle of the image
    fill_value = 127
    fill_offset = tile_size // 2
    # fmt: off
    data[
        fill_offset: (image_width - fill_offset),
        fill_offset: (image_width - fill_offset),
    ] = fill_value
    # fmt: on

    # generate the ground truth image
    suffix = ".ome.tiff"
    ground_truth_file = get_temp_file(ground_truth_dir, suffix)
    with bfio.BioWriter(ground_truth_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]

    stitching_vector = stitch_dir / ("img-global-positions" + ".txt")

    # stitching data
    offsets = [
        {"grid": (0, 0), "file": "img_r001_c001.ome.tif", "position": (0, 0)},
        {
            "grid": (0, 1),
            "file": "img_r001_c002.ome.tif",
            "position": (tile_size - offset_x, 0),
        },
        {
            "grid": (1, 0),
            "file": "img_r002_c001.ome.tif",
            "position": (0, tile_size - offset_y),
        },
        {
            "grid": (1, 1),
            "file": "img_r002_c002.ome.tif",
            "position": (tile_size - offset_x, tile_size - offset_y),
        },
    ]
    for offset in offsets:
        offset["corr"] = round(random.uniform(-1, 1), 10)  # type: ignore[arg-type]  # noqa: S311 E501

    # create stitching vector
    # TODO CHECK Filepattern updates.
    # A bug from filepattern prevents generating from dic for now
    stitching_data = [
        "file: img_r001_c001.ome.tif; corr: -0.0864568939; position: (0, 0); grid: (0, 0);",  # noqa: E501
        "file: img_r001_c002.ome.tif; corr: -0.657176744; position: (656, 0); grid: (0, 1);",  # noqa: E501
        "file: img_r002_c001.ome.tif; corr: 0.7119831612; position: (0, 1008); grid: (1, 0);",  # noqa: E501
        "file: img_r002_c002.ome.tif; corr: 0.2078192665; position: (656, 1008); grid: (1, 1);",  # noqa: E501
    ]

    with pathlib.Path.open(stitching_vector, "w") as f:
        for row in stitching_data:
            f.write(f"{row}\n")

    # TODO When Filepattern is fixed, generate the stitching vector like that
    # # create the stitching vector
    # with open(filename, "w") as f:
    #     for row in offsets:
    #         for key, value in row.items():

    # generate the partial images
    for offset in offsets:
        image_file: pathlib.Path = inp_dir.joinpath(str(offset["file"]))
        origin_x: int = offset["position"][0]  # type: ignore[assignment]
        origin_y: int = offset["position"][1]  # type: ignore[assignment]
        fov = data[
            origin_y : (origin_y + fov_height),
            origin_x : (origin_x + fov_width),
        ]
        with bfio.BioWriter(image_file) as writer:
            writer.X = fov_width
            writer.Y = fov_height
            writer[:] = fov


@pytest.fixture()
def gen_dirs() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    """Create temporary directories for testing."""

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("inp_dir")
    inp_dir.mkdir()

    out_dir = data_dir.joinpath("out_dir")
    out_dir.mkdir()

    stitch_dir = data_dir.joinpath("stitch_dir")
    stitch_dir.mkdir()

    ground_truth_dir = data_dir.joinpath("ground_truth_dir")
    ground_truth_dir.mkdir()

    gen_data(inp_dir, stitch_dir, ground_truth_dir)

    yield inp_dir, out_dir, stitch_dir, ground_truth_dir

    # Remove the directory after the test
    shutil.rmtree(data_dir)


@pytest.mark.skip("This test hangs.")
def test_image_assembler(
    gen_dirs: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Test correctness of the image assembler plugin in a basic case."""
    inp_dir, out_dir, stitch_dir, ground_truth_dir = gen_dirs

    assemble_images(inp_dir, stitch_dir, out_dir, False)

    assert len(os.listdir(ground_truth_dir)) == 1
    assert len(os.listdir(out_dir)) == 1

    ground_truth_file = ground_truth_dir / os.listdir(ground_truth_dir)[0]
    assembled_image_file = out_dir / os.listdir(out_dir)[0]

    # check assembled image against ground truth
    with (
        bfio.BioReader(ground_truth_file) as ground_truth,
        bfio.BioReader(assembled_image_file) as image,
    ):
        assert ground_truth.shape == image.shape
        assert (ground_truth[:] == image[:]).all()


def test_cli(
    gen_dirs: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Test the command line."""
    runner = typer.testing.CliRunner()

    inp_dir, out_dir, stitch_dir, _ = gen_dirs

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0


def test_cli_preview(
    gen_dirs: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Test the preview option."""
    runner = typer.testing.CliRunner()

    inp_dir, out_dir, stitch_dir, _ = gen_dirs

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
            "--preview",
        ],
    )

    print(result.exception)
    print(result.stdout)
    assert result.exit_code == 0

    with pathlib.Path.open(out_dir / "preview.json") as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    result = plugin_json["outputDir"]
    assert len(result) == 1
    assert pathlib.Path(result[0]).name == "img_r00(1-2)_c00(1-2).ome.tif"


def test_cli_bad_input(
    gen_dirs: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Test bad inputs."""
    runner = typer.testing.CliRunner()

    _, out_dir, stitch_dir, _ = gen_dirs
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exc_info[0] is ValueError
