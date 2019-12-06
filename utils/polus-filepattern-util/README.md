# **B**io**F**ormats **I**nput/**O**utput utility (bfio)

This tool is a simplified but powerful interface to the [Bioformats java library](https://www.openmicroscopy.org/bio-formats/). It makes use of Cell Profilers [python-bioformats](https://github.com/CellProfiler/python-bioformats) package to access the Bioformats library. One of the issues with using the `python-bioformats` package is reading and writing large image planes (>2GB). The challenge lies in the way Bioformats reads and writes large image planes, using an `int` value to index the file. To do get around this, files can be read or written in chunks and the classes provided in `bfio` handle this automatically. The `BioWriter` class in this package only writes files in the `.ome.tif` format, and automatically sets the tile sizes to 1024.

This tool is currently not on any public pip repositories, but can be installed by cloning this repository and installing with pip.

