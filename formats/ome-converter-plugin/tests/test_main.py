"""Testing of Ome Converter."""
import os
import pathlib
import shutil
import tempfile

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

EXT = [[".ome.tif", ".ome.zarr"]]


@pytest.fixture(params=EXT)
def extensions(request):
    """To get the parameter of the fixture."""
    return request.param


def test_batch_converter(extensions):
    """Create synthetic image."""
    arr = np.arange(0, 1048576, 1, np.uint8)
    arr = np.reshape(arr, (1024, 1024))
    image = Image.fromarray(arr)
    outname = "syn_image.png"
    syn_dir = tempfile.mkdtemp(dir=pathlib.Path.cwd())
    image.save(pathlib.Path(syn_dir, outname))
    br = BioReader(pathlib.Path(syn_dir, outname))
    assert br._backend_name == "java"
    for f in pathlib.Path(syn_dir).iterdir():
        for i in extensions:
            image = BioReader(f).read()
            tmp_dir = tempfile.mkdtemp(dir=pathlib.Path.cwd())
            out_file = pathlib.Path(tmp_dir, f.stem + i)
            with BioWriter(file_path=out_file) as bw:
                bw.X = image.shape[0]
                bw.Y = image.shape[1]
                bw.dtype = image.dtype
                bw[:] = image

            if i == ".ome.tif":
                fileExtension = ".ome.zarr"
            else:
                fileExtension = ".ome.tif"
            tmp2_dir = tempfile.mkdtemp(dir=pathlib.Path.cwd())
            batch_convert(
                pathlib.Path(tmp_dir), pathlib.Path(tmp2_dir), ".+", fileExtension
            )
            assert all([f for f in os.listdir(tmp_dir) if i in f]) is True

            shutil.rmtree(tmp_dir)
            shutil.rmtree(tmp2_dir)
    shutil.rmtree(syn_dir)


@pytest.fixture()
def images():
    """Download test /Users/abbasih2/Documents/Polus_Repos/polus-plugins/formats/ome-converter-plugin/data/inputimages."""
    inp_dir = tempfile.mkdtemp(dir=pathlib.Path.cwd())
    imagelist = {
        ("0.tif", "https://osf.io/j6aer/download"),
        (
            "img_r001_c001.ome.tif",
            "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/img_r001_c001.ome.tif",
        ),
        (
            "00001_01.ome.tiff",
            "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/MitoCheck/00001_01.ome.tiff",
        ),
    }
    out_files = []
    for im in imagelist:
        file, url = im
        r = requests.get(url)
        outfile = pathlib.Path(inp_dir, file)
        with open(outfile, "wb") as fw:
            fw.write(r.content)
            out_files.append(outfile)
    return out_files


def test_image_converter_omezarr(images, extensions):
    """Testing of bioformat supported image datatypes conversion to ome.zarr and ome.tif file format."""
    for im, i in zip(images, extensions):
        image = io.imread(im)
        out_dir = tempfile.mkdtemp(dir=pathlib.Path.cwd())
        out_file = pathlib.Path(out_dir, im.name.split(".")[0] + i)
        with BioWriter(file_path=out_file) as bw:
            bw.X = image.shape[1]
            bw.Y = image.shape[0]
            bw.dtype = image.dtype
            bw[:] = image
        if i == ".ome.tif":
            fileExtension = ".ome.zarr"
            backend_name = "zarr"
        else:
            fileExtension = ".ome.tif"
            backend_name = "python"
        out_dir2 = tempfile.mkdtemp(dir=pathlib.Path.cwd())
        out_file2 = pathlib.Path(out_dir2, out_file.name.split(".")[0] + fileExtension)
        convert_image(out_file, fileExtension, pathlib.Path(out_dir2))
        assert all([f for f in os.listdir(out_dir2) if fileExtension in f]) is True
        with BioReader(out_file2) as br:
            assert br._backend_name == backend_name

        shutil.rmtree(out_dir)
        shutil.rmtree(out_dir2)
    shutil.rmtree(images[0].parent)
