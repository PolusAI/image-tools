"""Testing of Ome Converter."""
import os
import pathlib
import sys

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(dirpath)


inpDir = pathlib.Path(dirpath, "data/input")
outDir = pathlib.Path(dirpath, "data/out")
if not outDir.exists():
    outDir.mkdir(exist_ok=True, parents=True)
filePattern = "p0{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif"
