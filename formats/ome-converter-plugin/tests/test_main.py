"""Testing of Ome Converter."""
import pathlib
import shutil
import tempfile
from collections.abc import Generator
from typing import Any, List, Tuple

import numpy as np
import pytest
import requests
import skimage
from bfio import BioReader
from skimage import io
from typer.testing import CliRunner

from polus.plugins.formats.ome_converter.__main__ import app as app
from polus.plugins.formats.ome_converter.image_converter import (
    batch_convert,
    convert_image,
)

runner = CliRunner()


EXT = [".ome.tif", ".ome.zarr"]


@pytest.fixture(params=EXT)
def file_extension(request):
    """To get the parameter of the fixture."""
    yield request.param


@pytest.fixture(
    params=[
        (256, ".png"),
        (512, ".tif"),
        (1024, ".png"),
        (2048, ".tif"),
        (4096, ".tif"),
    ]
)
def get_params(request):
    """To get the parameter of the fixture."""
    yield request.param


@pytest.fixture
def synthetic_images(
    get_params,
) -> Generator[Tuple[List[Any], pathlib.Path], None, None]:
    """Generate random synthetic images."""
    size, extension = get_params

    syn_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    images = []
    for i in range(10):
        # Create images
        blobs = skimage.data.binary_blobs(
            length=size, volume_fraction=0.05, blob_size_fraction=0.05
        )
        syn_img = skimage.measure.label(blobs)
        outname = f"syn_image_{i}{extension}"
        # Save image
        out_path = pathlib.Path(syn_dir, outname)
        io.imsave(out_path, syn_img)
        images.append(syn_img)

    yield images, syn_dir


@pytest.fixture
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Generate random synthetic images."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)


def test_batch_converter(synthetic_images, file_extension, output_directory) -> None:
    """Create synthetic image.

    This unit test runs the batch_converter and validates that the converted data is
    the same as the input data.
    """
    image, inp_dir = synthetic_images
    batch_convert(inp_dir, output_directory, ".+", file_extension)
    input_stems = {f.stem for f in inp_dir.iterdir()}
    output_stems = {f.name.split(".")[0] for f in output_directory.iterdir()}

    # Simple check to make sure all input files were converted
    assert input_stems == output_stems
    # # Check to make sure that output files are identical to original data
    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])
    shutil.rmtree(inp_dir)


@pytest.fixture
def images() -> Generator[pathlib.Path, None, None]:
    """Download test /Users/abbasih2/Documents/Polus_Repos/polus-plugins/formats/ome-converter-plugin/data/inputimages."""
    imagelist = {
        ("0.tif", "https://osf.io/j6aer/download/"),
        (
            "cameraman.png",
            "https://people.math.sc.edu/Burkardt/data/tif/cameraman.png",
        ),
        ("venus1.png", "https://people.math.sc.edu/Burkardt/data/tif/venus1.png"),
    }
    inp_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    for image in imagelist:
        file, url = image
        outfile = pathlib.Path(inp_dir, file)

        r = requests.get(url)
        with open(outfile, "wb") as fw:
            fw.write(r.content)

    yield outfile
    shutil.rmtree(inp_dir)


def test_image_converter_omezarr(images, file_extension, output_directory) -> None:
    """Testing of bioformat supported image datatypes conversion to ome.zarr and ome.tif file format."""
    br_img = BioReader(images)
    image = br_img.read()
    convert_image(images, file_extension, output_directory)

    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])


def test_cli(synthetic_images, output_directory, file_extension) -> None:
    """Test Cli."""
    _, inp_dir = synthetic_images

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".+",
            "--fileExtension",
            file_extension,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
