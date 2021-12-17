# Polus Tiled Tiff Conversion Plugin

This WIPP plugin takes any image type supported by Bioformats and converts it 
to an OME tiled tiff. The tiled storage format is helpful for loading and 
displaying huge images efficiently by loading only the necessary tiles (rather 
than loading a single, large image). For file formats with a pyramid-like 
structure with multiple resolutions (or series), this plugin only saves the 
first series to a tiff (usually, the first series is the highest resolution).

The current need for this plugin is that WIPP's tiled tiff conversion process
only grabs the first image plane, while this plugin grabs all image planes in a
series. Ultimately, this permits complete data conversion (including all
channels, z-positions, and time points). Each image plane (defined as a single
z-slice, channel, or time point) gets saved as a separate image. Suppose an 
image has multiple slices for a given dimension (z-slice, channel, or time 
point). In that case, the file name will include an indicator for the 
particular slice. For example, an image named `Image.czi` that has two 
z-slices, two channels, and two time points will have the following files 
exported by this plugin:

```bash
Image_z0_c0_t0.ome.tif
Image_z1_c0_t0.ome.tif
Image_z0_c1_t0.ome.tif
Image_z1_c1_t0.ome.tif
Image_z0_c0_t1.ome.tif
Image_z1_c0_t1.ome.tif
Image_z0_c1_t1.ome.tif
Image_z1_c1_t1.ome.tif
```

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build from source code, run `./mvn-packager.sh`

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description                   | I/O    | Type    | Required |
| ------------- | ----------------------------- | ------ | ------- | -------- |
| `input`       | Input image collection        | Input  | Path    | true     |
| `output`      | Output image collection       | Output | Path    | true     |

## Example Code

```Linux
mkdir examples
cd examples
mkdir output
wget -P images/ https://data.broadinstitute.org/bbbc/BBBC033/BBBC033_v1_dataset.zip
unzip images/BBBC033_v1_dataset.zip
basedir=$(basename ${PWD})
docker run -v ${PWD}:/$basedir labshare/polus-tiledtiff-converter-plugin:1.1.0 \
--input /$basedir/"images/" \
--output /$basedir/"output/"
```

Navigate to the `examples/output/` directory to visualize the tiles using a 
tool such as bfio.

## Viewing the results using Python

```Python
from bfio import BioReader
from pathlib import Path
import matplotlib.pyplot as plt

inpDir = Path("./images/")
outDir = Path("./output/")

input_files = [f for f in inpDir.iterdir() if f.is_file() and f.name.endswith('.tif')]
output_files = [f for f in inpDir.iterdir() if f.is_file() and f.name.endswith('.ome.tif')]

with BioReader(inpDir / input_files[0].name) as br_in:
    img_in = br_in[:]

with BioReader(outDir / output_files[0].name) as br_out:
    img_out = br_out[:]

fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].imshow(img_in), ax[0].set_title("Image 1")
ax[1].imshow(img_out), ax[1].set_title("Tile 1")
fig.suptitle(file.name)
plt.show()
```