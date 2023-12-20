"""Test fixtures.

Set up all data used in tests.
"""

import io
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import bfio
import numpy
import pytest
import requests
from bfio import BioWriter


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--downloads",
        action="store_true",
        dest="downloads",
        default=False,
        help="run tests that download large data files",
    )


@pytest.fixture()
def local_data() -> tuple[Path, Path, Path, Path]:  # type: ignore
    """Generate test data for local testing."""
    # create temporary directory for all data
    data_dir = Path(tempfile.mkdtemp(suffix="test_data"))

    img_path = data_dir / "img_dir"
    img_path.mkdir()

    stitch_path = data_dir / "stitch_dir"
    stitch_path.mkdir()

    output_path = data_dir / "output_dir"
    output_path.mkdir()

    ground_truth_path = data_dir / "ground_truth_dir"
    ground_truth_path.mkdir()

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
    data[
        fill_offset : (image_width - fill_offset),
        fill_offset : (image_width - fill_offset),
    ] = fill_value

    # generate the ground truth image
    ground_truth_file = ground_truth_path / "ground_truth.ome.tiff"
    with BioWriter(ground_truth_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]

    stitching_vector_file = stitch_path / "img-global-positions.txt"

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

    with Path.open(stitching_vector_file, "w") as f:
        for row in stitching_data:
            f.write(f"{row}\n")

    # TODO When Filepattern is fixed, generate the stitching vector like that
    # # create the stitching vector
    # with open(filename, "w") as f:
    #     for row in offsets:
    #         for key, value in row.items():

    # generate the partial images
    for offset in offsets:
        # NOTE LOOK AT THIS
        image_file: Path = img_path / offset["file"]  # type: ignore
        origin_x: int = offset["position"][0]  # type: ignore[assignment]
        origin_y: int = offset["position"][1]  # type: ignore[assignment]
        fov = data[
            origin_y : (origin_y + fov_height),
            origin_x : (origin_x + fov_width),
        ]
        with BioWriter(image_file) as writer:
            writer.X = fov_width
            writer.Y = fov_height
            writer[:] = fov

    yield (img_path, stitch_path, output_path, ground_truth_path)

    # remove the temporary directory
    shutil.rmtree(data_dir)


@pytest.fixture()
def nist_data() -> tuple[Path, Path, Path]:  # type: ignore
    """Download the NIST MIST dataset."""
    http_request_timeout = 10

    data_dir = Path(tempfile.mkdtemp(suffix="test_data"))

    # download the FOVs
    nist_raw_dir = data_dir / "nist-raw-dir"
    if not nist_raw_dir.exists():
        r = requests.get(
            url="https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset.zip",
            timeout=http_request_timeout,
        )
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(nist_raw_dir)

    assert nist_raw_dir.exists(), "Could not download nist images"

    img_raw_dir = nist_raw_dir / "Small_Phase_Test_Dataset" / "image-tiles"
    assert img_raw_dir.exists(), "downloaded images are malformed"

    # download the stitching vector
    stitch_raw_dir = data_dir / "stitch-raw-dir"
    if not stitch_raw_dir.exists():
        r = requests.get(
            url="https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset_Example_Results.zip",
            timeout=http_request_timeout,
        )
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(stitch_raw_dir)

    assert stitch_raw_dir.exists(), "Could not download stitching vector"

    stitch_raw_dir = stitch_raw_dir / "Small_Phase_Test_Dataset_Example_Results"
    assert stitch_raw_dir.exists(), "downloaded stitching vector is malformed"

    img_dir = data_dir / "img-dir"
    img_dir.mkdir(exist_ok=True)

    # Convert the images into ome.tif
    for img in img_raw_dir.iterdir():
        if img.suffix == ".tif":
            out_path = img_dir / f"{img.stem}.ome.tif"
            with bfio.BioReader(img) as reader, bfio.BioWriter(
                out_path,
                metadata=reader.metadata,
            ) as writer:
                image = reader[:].squeeze().astype(numpy.float32)
                writer.Y = image.shape[0]
                writer.X = image.shape[1]
                writer.dtype = image.dtype
                writer[:] = image

    stitch_path = data_dir / "stitch_dir"
    stitch_path.mkdir(exist_ok=True)

    # Update the names of image files in stitching vector
    for vector in stitch_raw_dir.iterdir():
        if vector.name == "img-global-positions-0.txt":
            out_vec = stitch_path / vector.name
            with vector.open("r") as reader, out_vec.open("w") as writer:
                for line in reader.readlines():
                    if ".tif" in line:
                        writer.write(line.replace(".tif", ".ome.tif"))
                    else:
                        writer.write(line)

    out_dir = data_dir / "out-dir"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    if not stitch_path.exists():
        msg = "could not successfully download nist_mist_dataset stitching vector"
        raise FileNotFoundError(msg)

    yield (img_dir, stitch_path, out_dir)

    # remove the temporary directory
    shutil.rmtree(data_dir)
