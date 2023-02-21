"""Testing of Ome Converter."""
# import os
# import pathlib
# import sys
# from multiprocessing import cpu_count

# import filepattern as fp
# from bfio import BioReader

# from src.polus.plugins.formats.ome_converter.image_converter import \
#     image_converter

# dirpath = os.path.abspath(os.path.join(__file__, "../.."))
# sys.path.append(dirpath)


# inpDir = pathlib.Path(dirpath, "data/inputs")
# outDir = pathlib.Path(dirpath, "data/out")
# if not outDir.exists():
#     outDir.mkdir(exist_ok=True, parents=True)
# filePattern = "p0{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif"


# def test_image_converter_omezarr():
#     """Testing of bioformat supported image datatypes conversion to ome.zarr file format."""
#     fileExtension = ".ome.zarr"
#     pattern = ".*"
#     fps = fp.FilePattern(inpDir, pattern)
#     for file in fps():
#         fl = file[1][0]
#         image_converter(pathlib.Path(fl), fileExtension, outDir)

#     assert all([f for f in os.listdir(outDir) if fileExtension in f]) is True


# def test_image_converter_ometif():
#     """Test of bioformat supported image datatypes conversion to ome.tif file format."""
#     fileExtension = ".ome.tif"
#     pattern = ".*"
#     omedir = pathlib.Path(dirpath, "data/ome")
#     if not omedir.exists():
#         omedir.mkdir(exist_ok=True, parents=True)
#     fps = fp.FilePattern(outDir, pattern)
#     for file in fps():
#         fl = file[1][0]
#         image_converter(pathlib.Path(fl), fileExtension, omedir)

#     assert all([f for f in os.listdir(outDir) if fileExtension in f]) is True


# def test_bfio_backend():
#     """Testing of bfio backend when reading images."""
#     pattern = ".*"
#     fps = fp.FilePattern(inpDir, pattern)
#     for file in fps():
#         fl = pathlib.Path(file[1][0])
#         with BioReader(pathlib.Path(fl), max_workers=cpu_count()) as br:
#             assert br._backend_name == "python"

#     fps = fp.FilePattern(outDir, pattern)
#     for file in fps():
#         fl = pathlib.Path(file[1][0])
#         with BioReader(pathlib.Path(fl), max_workers=cpu_count()) as br:
#             assert br._backend_name == "zarr"
