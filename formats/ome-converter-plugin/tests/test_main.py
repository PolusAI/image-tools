"""Testing of Ome Converter."""
import os
import pathlib
import sys
from multiprocessing import cpu_count

import filepattern as fp
import numpy as np
import requests
from bfio import BioReader, BioWriter
from PIL import Image
from skimage import io

from polus.plugins.formats.ome_converter.image_converter import image_converter

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(dirpath)

inpDir = pathlib.Path(dirpath, "data/input")
outDir = pathlib.Path(dirpath, "data/out")
omeDir = pathlib.Path(dirpath, "data/outome")
zarrDir = pathlib.Path(dirpath, "data/outzarr")
if not inpDir.exists():
    inpDir.mkdir(parents=True, exist_ok=True)
if not outDir.exists():
    outDir.mkdir(exist_ok=True, parents=True)
if not omeDir.exists():
    omeDir.mkdir(exist_ok=True, parents=True)
if not zarrDir.exists():
    zarrDir.mkdir(parents=True, exist_ok=True)

imagelist = {
    "0.tif": "https://osf.io/j6aer/download",
    "img_r001_c001.ome.tif": "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/img_r001_c001.ome.tif",
    "00001_01.ome.tiff": "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/MitoCheck/00001_01.ome.tiff",
}
pattern = ".*"


def download_images():
    """Download test /Users/abbasih2/Documents/Polus_Repos/polus-plugins/formats/ome-converter-plugin/data/inputimages."""
    for file, url in imagelist.items():
        r = requests.get(url)
        with open(inpDir.joinpath(file), "wb") as fw:
            fw.write(r.content)


def generate_sythetic_images():
    """Generate synthetic images."""
    arr = np.arange(0, 1048576, 1, np.uint8)
    arr = np.reshape(arr, (1024, 1024))
    image = Image.fromarray(arr)
    image.save(inpDir.joinpath("syn_image_1.png"))


def omewrite(image, outfile):
    """Write images to using bfiowriter."""
    with BioWriter(file_path=outfile) as bw:
        bw.X = image.shape[0]
        bw.Y = image.shape[1]
        bw.dtype = image.dtype
        bw[:] = image


def test_image_converter_omezarr():
    """Testing of bioformat supported image datatypes conversion to ome.zarr file format."""
    download_images()
    fname = inpDir.joinpath(list(imagelist.keys())[0])
    image = io.imread(fname)
    outfile = outDir.joinpath(fname.name.split(".")[0] + ".ome.tif")
    omewrite(image, outfile)
    fileExtension = ".ome.zarr"
    fps = fp.FilePattern(outDir, pattern)
    for file in fps():
        fl = file[1][0]
        image_converter(pathlib.Path(fl), ".ome.zarr", zarrDir)
    assert all([f for f in os.listdir(outDir) if fileExtension in f]) is True


def test_image_converter_ometif():
    """Test of bioformat supported image datatypes conversion to ome.tif file format."""
    fileExtension = ".ome.tif"
    fps = fp.FilePattern(outDir, pattern)
    for file in fps():
        fl = file[1][0]
        image_converter(pathlib.Path(fl), fileExtension, omeDir)
    assert all([f for f in os.listdir(outDir) if fileExtension in f]) is True


def test_bfio_backend():
    """Testing of bfio backend when reading images."""
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

    br = BioReader(inpDir.joinpath(list(imagelist.keys())[2]))
    assert br._backend_name != "python"

    generate_sythetic_images()
    br = BioReader(inpDir.joinpath("syn_image_1.png"))
    assert br._backend_name == "java"
