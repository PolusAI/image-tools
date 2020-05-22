# **B**io**F**ormats **I**nput/**O**utput utility (bfio)

This tool is a simplified but powerful interface to the [Bioformats java library](https://www.openmicroscopy.org/bio-formats/). It makes use of Cell Profilers [python-bioformats](https://github.com/CellProfiler/python-bioformats) package to access the Bioformats library. One of the issues with using the `python-bioformats` package is reading and writing large image planes (>2GB). The challenge lies in the way Bioformats reads and writes large image planes, using an `int` value to index the file. To do get around this, files can be read or written in chunks and the classes provided in `bfio` handle this automatically. The `BioWriter` class in this package only writes files in the `.ome.tif` format, and automatically sets the tile sizes to 1024.

This tool is currently not on any public pip repositories, but can be installed by cloning this repository and installing with pip.

## Universal Container Components

All containers contain the follow components:
1. Python 3.6
2. openjdk-8
3. numpy (version 1.18.1)
4. javabridge (version 1.0.18)
5. python-bioformats (version 1.5.2)
6. bfio (version 1.0.8)

## Containers

### labshare/polus-bfio-util:1.0.8 & labshare/polus-bfio-util:1.0.8-alpine

This container is built on Alpine Linux. This is the smallest bfio container, but also the most difficult to install additional requirements on.

### labshare/polus-bfio-util:1.0.8-slim-buster

This container is built on a stripped down version of Debian Buster. This container is larger than the `alpine` version, but easier to install new Python packages on.

### labshare/polus-bfio-util:1.0.8-tensorflow

This container is built on Debian Buster and includes Tensorflow 2.1.0 and all necessary GPU drivers to run Tensorflow on an NVIDIA graphics card.