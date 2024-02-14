# FTL Label (v0.3.12-dev5)

This plugin performs a transformation on binary images which, in a certain limiting case, can be thought of as segmentation.

This plugin is an n-dimensional connected component algorithm that is similar to the [Light Speed Labeling](http://www-soc.lip6.fr/~lacas/Publications/ICIP09_LSL.pdf) algorithm.
As mentioned in the Wikipedia article, (Connected-component labeling)
[https://en.wikipedia.org/wiki/Connected-component_labeling] should not
necessarily be thought of as segmentation.
The LSL algorithm uses a novel relative (rather than absolute) line labeling
scheme to minimize the number of conditional statements and thus CPU pipeline
stalls. The relative labeling introduces an additional pass, and thus LSL is a
three-pass algorithm.

This algorithm works in n-dimensions and uses run length encoding to compress the image and accelerate computation.
As a reference to the Light Speed Labeling algorithm, we named this method the Faster Than Light (FTL) Labeling algorithm, although this should not be interpreted as this algorithm being faster.

The `Cython` implementation generally performs better than [SciKit's `label` method](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label), except on images with random noise in which it performs up to 10x slower and uses 2x more memory.
In most of our real test images, this algorithm ran 2x faster and used 4x less memory.
This implementation does load the entire image into memory and, so, is not suitable for extremely large images.

The `Rust` implementation processes the images in tiles rather than all at once.
This lets it scale to arbitrarily large sizes but does make it slower than the Cython implementation.
However, most of the bottleneck is in the interface between `Python` and `Rust`.
The Rust implementation works with 2d and 3d images.

To see detailed documentation for the `Rust` implementation you need to:

* Install [Rust](https://doc.rust-lang.org/stable/book/ch01-01-installation.html),
* add Cargo to your `PATH`, and
* run from the terminal (in this directory): `cargo doc --open`.

That last command will generate documentation and open a new tab in your default web browser.

We determine whether to use the `Cython` or `Rust` implementation on a per-image basis depending on the size of that image.
If we expect the image to occupy less than `500MB` of memory, we use the `Cython` implementation otherwise we use the `Rust` implementation.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## To do

The following optimizations should be added to increase the speed or decrease the memory used by the plugin.

1. Implement existing specialized C++ methods that accelerate the run length encoding operation by a factor of 5-10

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name                      | Description                                                          | I/O    | Type       |
| ------------------------- | -------------------------------------------------------------------- | ------ | ---------- |
| `--inpDir`                | Input image collection to be processed by this plugin                | Input  | collection |
| `--connectivity`          | City block connectivity                                              | Input  | number     |
| `--binarizationThreshold` | For images containing probability values. Must be between 0 and 1.0. | Input  | number     |
| `--outDir`                | Output collection                                                    | Output | collection |

## Example Code

```Linux
# Download some example *.tif files
wget https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
unzip dsb2018.zip
mv dsb2018/test/masks/ images/

# Convert the *.tif files to *.ome.tif tiled tif format using bfio.
basedir=$(basename ${PWD})
docker run -v ${PWD}:/$basedir labshare/polus-tiledtiff-converter-plugin:1.1.0 \
  --input /$basedir/images/ \
  --output /$basedir/images_ome/

# Run the FTL label plugin
mkdir output
docker run -v ${PWD}:/$basedir labshare/polus-ftl-label-plugin:0.3.10 \
--inpDir /$basedir/"images_ome/" \
--outDir /$basedir/"output/" \
--connectivity 1

# View the results using bfio and matplotlib
# Let's run directly on the host since we just need the python backend.
pip install bfio==2.1.9 matplotlib==3.5.1
python3 SimpleTiledTiffViewer.py --inpDir images_ome/ --outDir output/
```

**NOTE:**
Connectivity uses [SciKit's](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label) notation for connectedness, which we call cityblock notation.
As you increase the connectivity, you increase the number of pixel jumps away from the center point.
For example, in 2D there are 4 neighbors using 1-connectivity and 8 neighbors using 2-connectivity,
whereas in 3D there are 6 neighbors using 1-connectivity and 26 neighbors using 2-connectivity.
Each new jump must be orthogonal to all previous jumps.
This means that `connectivity` should have a minimum value of `1` and a maximum value equal to the dimensionality of the images.

SciKit's documentation has a good illustration for 2D:

```text
1-connectivity     2-connectivity     diagonal connection close-up

     [ ]           [ ]  [ ]  [ ]             [ ]
      |               \  |  /                 |  <- hop 2
[ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
      |               /  |  \             hop 1
     [ ]           [ ]  [ ]  [ ]
```
