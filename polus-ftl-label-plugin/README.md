# FTL Label

This plugin is an n-dimensional connected component algorithm that is similar to
the
[Light Speed Labeling](http://www-soc.lip6.fr/~lacas/Publications/ICIP09_LSL.pdf)
algorithm. This algorithm works in n-dimensions and uses run length encoding to
compress the image and accelerate computation. As a reference to the Light Speed
Labeling algorithm, this method was named the Faster Than Light (FTL) Labeling
algorithm, although this should not be interpreted as this algorithm being faster.
This algorithm generally performs better than
[SciKit's `label` method](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label),
except on images with random noise in which it performs up to 10x slower and
uses 2x more memory. In most of our real test images, this algorithm ran 2x
faster and used 4x less memory.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## To do

The following optimizations should be added to increase the speed or decrease
the memory used by the plugin.
1. Implement existing specialized C++ methods that accelerate the run length encoding operation by a factor of 5-10
2. Create a Cython class that can process strips of images as they are loaded into memory.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name             | Description                                           | I/O    | Type       |
|------------------|-------------------------------------------------------|--------|------------|
| `--connectivity` | City block connectivity                               | Input  | number     |
| `--inpDir`       | Input image collection to be processed by this plugin | Input  | collection |
| `--outDir`       | Output collection                                     | Output | collection |

**NOTE:** Connectivity uses
[SciKit's](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label)
notation for connectedness, which is called cityblock notation here. For 2D,
1-connectivity is the same as 4-connectivity and in 3D is the same as
6-connectivity. As you increase the connectivity, you increase the number of
pixel jumps away from the center point. SciKit's documentation has a good
illustration for 2D:
```
1-connectivity     2-connectivity     diagonal connection close-up

     [ ]           [ ]  [ ]  [ ]             [ ]
      |               \  |  /                 |  <- hop 2
[ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
      |               /  |  \             hop 1
     [ ]           [ ]  [ ]  [ ]
```