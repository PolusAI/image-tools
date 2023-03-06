"""Testing of Ome Converter."""
import os
import pathlib
import sys
from multiprocessing import cpu_count

import filepattern as fp
import numpy as np
import pytest
import requests
from bfio import BioReader, BioWriter
from PIL import Image
from skimage import io

from polus.plugins.formats.ome_converter.image_converter import (
    batch_convert,
    convert_image,
)

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(dirpath)

inpDir = pathlib.Path(dirpath, "data/input")
outDir = pathlib.Path(dirpath, "data/out")
omeDir = pathlib.Path(dirpath, "data/outome")
zarrDir = pathlib.Path(dirpath, "data/outzarr")
synDir = pathlib.Path(dirpath, "data/synDir")
if not inpDir.exists():
    inpDir.mkdir(parents=True, exist_ok=True)
if not outDir.exists():
    outDir.mkdir(exist_ok=True, parents=True)
if not omeDir.exists():
    omeDir.mkdir(exist_ok=True, parents=True)
if not zarrDir.exists():
    zarrDir.mkdir(parents=True, exist_ok=True)
if not synDir.exists():
    synDir.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def dowload_images():
    """Download test /Users/abbasih2/Documents/Polus_Repos/polus-plugins/formats/ome-converter-plugin/data/inputimages."""
    imagelist = {
        "0.tif": "https://osf.io/j6aer/download",
        "img_r001_c001.ome.tif": "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/img_r001_c001.ome.tif",
        "00001_01.ome.tiff": "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/MitoCheck/00001_01.ome.tiff",
    }

    for file, url in imagelist.items():
        r = requests.get(url)
        with open(inpDir.joinpath(file), "wb") as fw:
            fw.write(r.content)

    return imagelist


@pytest.fixture
def omewrite_func():
    """Write images to using bfiowriter."""

    def _omewrite(image, outfile):
        with BioWriter(file_path=outfile) as bw:
            bw.X = image.shape[0]
            bw.Y = image.shape[1]
            bw.dtype = image.dtype
            bw[:] = image

    return _omewrite


def test_batch_converter(omewrite_func):
    """Create synthetic images."""
    pattern = ".+"
    arr = np.arange(0, 1048576, 1, np.uint8)
    arr1 = np.arange(0, 262144, 1, np.uint8)
    arr = np.reshape(arr, (1024, 1024))
    arr1 = np.reshape(arr1, (512, 512))
    image = Image.fromarray(arr)
    image1 = Image.fromarray(arr1)
    outname = "syn_image.png"
    outname1 = "syn_image_1.png"
    image.save(synDir.joinpath(outname))
    image1.save(synDir.joinpath(outname1))
    for f in synDir.iterdir():
        image = BioReader(f).read()
        print(synDir.joinpath(f.stem + ".ome.tif"))
        omewrite_func(image, omeDir.joinpath(f.stem + ".ome.tif"))

    for f in synDir.iterdir():
        image = BioReader(f).read()
        print(synDir.joinpath(f.stem + ".ome.zarr"))
        omewrite_func(image, zarrDir.joinpath(f.stem + ".ome.zarr"))

    batch_convert(omeDir, zarrDir, pattern, ".ome.zarr")
    batch_convert(zarrDir, omeDir, pattern, ".ome.tif")
    assert all([f for f in os.listdir(omeDir) if ".ome.tif" in f]) is True
    assert all([f for f in os.listdir(zarrDir) if ".ome.zarr" in f]) is True


def test_image_converter_omezarr(dowload_images, omewrite_func):
    """Testing of bioformat supported image datatypes conversion to ome.zarr and ome.tif file format."""
    fname = inpDir.joinpath(list(dowload_images.keys())[0])
    image = io.imread(fname)
    outfile = outDir.joinpath(fname.name.split(".")[0] + ".ome.tif")
    omewrite_func(image, outfile)
    pattern = ".+"
    fileExtension = ".ome.zarr"
    fps = fp.FilePattern(outDir, pattern)
    for file in fps():
        fl = file[1][0]
        convert_image(pathlib.Path(fl), fileExtension, zarrDir)
    assert all([f for f in os.listdir(zarrDir) if fileExtension in f]) is True

    fileExtension = ".ome.tif"
    fps = fp.FilePattern(zarrDir, pattern)
    for file in fps():
        fl = file[1][0]
        convert_image(pathlib.Path(fl), fileExtension, omeDir)
    assert all([f for f in os.listdir(omeDir) if fileExtension in f]) is True


def test_bfio_backend(dowload_images):
    """Testing of bfio backend when reading images."""
    pattern = ".*"
    fps = fp.FilePattern(omeDir, pattern)
    for file in fps():
        fl = pathlib.Path(file[1][0])
        with BioReader(pathlib.Path(fl), max_workers=cpu_count()) as br:
            assert br._backend_name == "python"

    fps = fp.FilePattern(zarrDir, pattern)
    for file in fps():
        fl = pathlib.Path(file[1][0])
        with BioReader(pathlib.Path(fl), max_workers=cpu_count()) as br:
            assert br._backend_name == "zarr"

    br = BioReader(inpDir.joinpath(list(dowload_images.keys())[2]))
    assert br._backend_name != "python"

    br = BioReader(synDir.joinpath("syn_image_1.png"))
    assert br._backend_name == "java"
