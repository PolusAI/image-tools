# FTL Label (v1.0.0-dev0)

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

We determine whether to use the `Cython` or `Rust` implementation on a per-image basis depending on the size of that image. If we expect the image to occupy less than `500MB` of memory, we use the `Cython` implementation otherwise we use the `Rust` implementation.
On macOS ARM64 (Apple Silicon), all images are automatically routed through the Rust backend.


To see detailed documentation for the `Rust` implementation you need to:

* Install [Rust](https://doc.rust-lang.org/stable/book/ch01-01-installation.html),
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

## Installation
#### From source (recommended for development)
```bash
# 1. Clone the repo
git clone <repo-url>
cd ftl-label-tool

# 2. Create a virtual environment
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package (compiles Cython + Rust extensions)
uv pip install -e .

# 4. Install the package with optional dependencies
uv pip install ".[dev]"

```

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

## Usage
```bash
python -m polus.images.transforms.images.ftl_label \
    --inpDir /path/to/input \
    --outDir /path/to/output \
    --connectivity 1 \
    --binarizationThreshold 0.5
```

## Docker
#### Building
To build the Docker image for the conversion plugin, run `./build-docker.sh`.

#### Run

```bash
basedir=$(basename ${PWD})

docker run -v ${PWD}:/$basedir polusai/ftl-label-tool:0.3.12 \
    --inpDir /$basedir/images/ \
    --outDir /$basedir/output/ \
    --connectivity 1 \
    --binarizationThreshold 0.5
```

## Example

```bash

# Run FTL label

python -m polus.images.transforms.images.ftl_label \
    --inpDir /path/to/images/ \
    --outDir /path/to/output/  \
    --connectivity 1 \
    --binarizationThreshold 0.5
```


**Connectivity:**
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


## Rust documentation
To generate and view the full Rust API docs:

```bash
cargo doc --open
```

## To Do
The following optimizations should be added to increase the speed or decrease the memory used by the plugin.

1. Implement existing specialized C++ methods that accelerate the run length encoding operation by a factor of 5-10

## For more information
To generate and view the full Rust API docs:

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).
