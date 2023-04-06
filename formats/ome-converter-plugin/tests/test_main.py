"""Testing of Ome Converter."""
import pathlib
import shutil
import tempfile
from collections.abc import Generator
from typing import Any, List, Tuple

import numpy as np
import pytest
import requests
from bfio import BioReader
from numpy import asarray
from PIL import Image

from polus.plugins.formats.ome_converter.image_converter import (
    batch_convert,
    convert_image,
)

EXT = [".ome.tif", ".ome.zarr"]


@pytest.fixture(params=EXT)
def file_extension(request):
    """To get the parameter of the fixture."""
    yield request.param


@pytest.fixture
def synthetic_pngs() -> Generator[Tuple[List[Any], pathlib.Path], None, None]:
    """Generate random synthetic images."""
    syn_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    images = []
    for i in range(10):
        # Create images
        arr = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        # Conver to PIL
        image = Image.fromarray(arr)
        # Save as PNG in tmp directory
        outname = f"syn_image_{i}.png"
        out_path = pathlib.Path(syn_dir, outname)
        image.save(out_path)
        images.append(arr)

    yield images, syn_dir
    shutil.rmtree(syn_dir)


@pytest.fixture
def synthetic_rgb() -> Generator[Tuple[List[Any], pathlib.Path], None, None]:
    """Generate random synthetic RGB images."""
    syn_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    images = []
    for i in range(10):
        image = Image.new(mode="RGB", size=(1024, 1024), color=(153, 153, 255))
        arr = asarray(image)
        outname = f"syn_image_{i}.tif"
        out_path = pathlib.Path(syn_dir, outname)
        image.save(out_path)
        images.append(arr)

    yield images, syn_dir
    shutil.rmtree(syn_dir)


@pytest.fixture
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Generate random synthetic images."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)


def test_batch_converter(synthetic_pngs, file_extension, output_directory) -> None:
    """Create synthetic image.

    This unit test runs the batch_converter and validates that the converted data is
    the same as the input data.
    """
    image, inp_dir = synthetic_pngs
    batch_convert(inp_dir, output_directory, ".+", file_extension)
    input_stems = {f.stem for f in inp_dir.iterdir()}
    output_stems = {f.name.split(".")[0] for f in output_directory.iterdir()}

    # Simple check to make sure all input files were converted
    assert input_stems == output_stems
    # # Check to make sure that output files are identical to original data
    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])


@pytest.mark.xfail
def test_batch_converter_rgb(synthetic_rgb, file_extension, output_directory) -> None:
    """Create synthetic image.

    This unit test fails as bfio.BioWriter only write single channel image but these are RGB images.
    """
    image, inp_dir = synthetic_rgb
    batch_convert(inp_dir, output_directory, ".+", file_extension)
    input_stems = {f.stem for f in inp_dir.iterdir()}
    output_stems = {f.name.split(".")[0] for f in output_directory.iterdir()}

    # Simple check to make sure all input files were converted
    assert input_stems == output_stems
    # # Check to make sure that output files are identical to original data
    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])


@pytest.fixture()
def images() -> Generator[pathlib.Path, None, None]:
    """Download test /Users/abbasih2/Documents/Polus_Repos/polus-plugins/formats/ome-converter-plugin/data/inputimages."""
    inp_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    imagelist = {
        ("0.tif", "https://osf.io/j6aer/download/"),
        (
            "cameraman.png",
            "https://people.math.sc.edu/Burkardt/data/tif/cameraman.png",
        ),
        ("venus1.png", "https://people.math.sc.edu/Burkardt/data/tif/venus1.png"),
    }
    for image in imagelist:
        file, url = image
        r = requests.get(url)
        outfile = pathlib.Path(inp_dir, file)
        with open(outfile, "wb") as fw:
            fw.write(r.content)
    yield outfile
    shutil.rmtree(inp_dir)


def test_image_converter_omezarr(images, file_extension, output_directory) -> None:
    """Testing of bioformat supported image datatypes conversion to ome.zarr and ome.tif file format."""
    image_fname = images
    br_img = BioReader(image_fname)
    image = br_img.read()
    convert_image(image_fname, file_extension, output_directory)
    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])
