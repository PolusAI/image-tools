# **B**io**F**ormats **I**nput/**O**utput utility (bfio)

This tool is a simplified but powerful interface to the
[Bioformats java library](https://www.openmicroscopy.org/bio-formats/).
It makes use of Cell Profilers
[python-bioformats](https://github.com/CellProfiler/python-bioformats)
package to access the Bioformats library. One of the issues with using the
`python-bioformats` package is reading and writing large image planes (>2GB).
The challenge lies in the way Bioformats reads and writes large image planes,
using an `int` value to index the file. To get around this, files can be read or
written in chunks and the classes provided in `bfio` handle this automatically.
The `BioWriter` class in this package only writes files in the `.ome.tif`
format, and automatically sets the tile sizes to 1024.

Docker containers with all necessary components are available (see
**Docker Containers** section).

## Documentation

Documentation is available on
[Read the Docs](https://bfio.readthedocs.io/en/latest/).

## Universal Container Components

All containers contain the follow components:

1. Python 3.8
2. [numpy](https://pypi.org/project/numpy/1.19.1/) (1.19.1)
3. [imagecodecs](https://pypi.org/project/imagecodecs/2020.5.30/) (2020.5.30, built with `--lite` option)
4. [tifffile](https://pypi.org/project/tifffile/2020.7.4/) (2020.7.4)
5. bfio (version 2.0.6)

Containers ending with `-java` also contain:

6. openjdk-8
7. [python-javabridge](https://pypi.org/project/python-javabridge/4.0.0/) (version 4.0.0)
8. [python-bioformats](https://pypi.org/project/python-bioformats/4.0.0/) (version 4.0.0)
9. [loci-tools.jar](https://downloads.openmicroscopy.org/bio-formats/6.1.0/artifacts/) (Version 6.1.0)

## Docker Containers

All containers can use the Python backend, but only the containers with Java may
use the Java backend. 

### ~labshare/polus-bfio-util:2.0.6~

The alpine container for `bfio` is currently unavailable.

~*Additional Python container tags:* `2.0.6-alpine`, `2.0.6-python`,~
~`2.0.6-alpine-python`~

~*Containers with Java:* `2.0.6-java`, `2.0.6-alpine-java`~

~This container is built on Alpine Linux. This is the smallest bfio container,~
~but also the most difficult to install additional requirements on. The Python~
~containers (98MB) are much smaller than the Java containers (383MB).~

### labshare/polus-bfio-util:2.0.6-slim-buster

*Additional Python container tags:* `2.0.6-slim-buster-python`

*Containers with Java:* `2.0.6-slim-buster-java`

This container is built on a stripped down version of Debian Buster. This
container is larger than the `alpine` version, but easier to install new Python
packages on since `manylinux` wheels can be installed on it. However, if a
package requires compilation, a compiler will need to be installed.

### labshare/polus-bfio-util:2.0.6-tensorflow

*Additional Python container tags:* `2.0.6-tensorflow-python`

*Containers with Java:* No Java containers

This container is built on Debian Buster and includes Tensorflow 2.1.0 and all
necessary GPU drivers to run Tensorflow on an NVIDIA graphics card.