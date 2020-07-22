# **B**io**F**ormats **I**nput/**O**utput utility (bfio)

This tool is a simplified but powerful interface to the [Bioformats java library](https://www.openmicroscopy.org/bio-formats/). It makes use of Cell Profilers [python-bioformats](https://github.com/CellProfiler/python-bioformats) package to access the Bioformats library. One of the issues with using the `python-bioformats` package is reading and writing large image planes (>2GB). The challenge lies in the way Bioformats reads and writes large image planes, using an `int` value to index the file. To do get around this, files can be read or written in chunks and the classes provided in `bfio` handle this automatically. The `BioWriter` class in this package only writes files in the `.ome.tif` format, and automatically sets the tile sizes to 1024.

Docker containers with all necessary components are available (see **Docker Containers** section).

## Examples

### Read an Image

```python
import javabridge
from bfio import BioReader,BioWriter,JARS
from pathlib import Path

# Import javabridge and start the vm
javabridge.start_vm(class_path=JARS)

# Path to bioformats supported image
image_path = Path('path/to/file.ome.tif')

# Create the BioReader object
bf = BioReader(str(image_path))

# Load the full image
image = bf.read_image()

# Only load the first 256x256 pixels, will still load all Z,C,T dimensions
# Note: Images are always 5-dimensional
image = bf.read_image(X=(0,256),Y=(0,256))

# Only load the second channel
image = bf.read_image(C=[1])

# Done executing program, so kill the vm. If the program needs to be run
# again, a new interpreter will need to be spawned to start the vm.
javabridge.kill_vm()
```

### Write an Image

```python
import javabridge
from bfio import BioReader,BioWriter,JARS
from pathlib import Path

# Import javabridge and start the vm
javabridge.start_vm(class_path=JARS)

# Path to bioformats supported image
image_path = Path('path/to/file.ome.tif')

# Create the BioReader object
br = BioReader(str(image_path))

# Load the full image
image = br.read_image()

# Create an image write object, rename the channels
bw = BioWriter(str(image_path.with_name("New_" + image_path.name)),image=image)
bw.channel_names(["Empty","ZO1","Empty"])
bw.write_image(image)
bw.close_image()

# Only save one channel
bw = BioWriter(str(image_path.with_name("New_" + image_path.name)),image=image)
bw.num_c(1)
bw.write_image(image[:,:,0,1,0].reshape((image.shape[0],image.shape[1],1,1,1)))
bw.close_image()

# List the channel names
print(bw.channel_names())

# Done executing program, so kill the vm. If the program needs to be run
# again, a new interpreter will need to be spawned to start the vm.
jutil.kill_vm()
```

## Universal Container Components

All containers contain the follow components:
1. Python 3.6
2. openjdk-8
3. numpy (version 1.18.1)
4. javabridge (version 1.0.18)
5. python-bioformats (version 1.5.2)
6. bfio (version 1.3.5)
7. [loci-tools.jar](https://downloads.openmicroscopy.org/bio-formats/6.1.0/artifacts/) (Version 6.1.0)

## Docker Containers

### labshare/polus-bfio-util:1.3.5 & labshare/polus-bfio-util:1.3.5-alpine

This container is built on Alpine Linux. This is the smallest bfio container, but also the most difficult to install additional requirements on.

### labshare/polus-bfio-util:1.3.5-slim-buster

This container is built on a stripped down version of Debian Buster. This container is larger than the `alpine` version, but easier to install new Python packages on.

### labshare/polus-bfio-util:1.3.5-tensorflow

This container is built on Debian Buster and includes Tensorflow 2.1.0 and all necessary GPU drivers to run Tensorflow on an NVIDIA graphics card.