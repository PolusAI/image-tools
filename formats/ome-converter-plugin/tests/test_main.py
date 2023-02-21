"""Testing of Ome Converter."""
import os
import pathlib
import sys

import filepattern as fp

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(dirpath)

inpDir = pathlib.Path(dirpath, "data/input")
outDir = pathlib.Path(dirpath, "data/out")
if not outDir.exists():
    outDir.mkdir(exist_ok=True, parents=True)
filePattern = "p0{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif"


def test_image_converter_omezarr():
    """Testing of bioformat supported image datatypes conversion to ome.zarr file format."""
    pattern = ".*"
    fps = fp.FilePattern(inpDir, pattern)
    for file in fps():
        file[1][0]
