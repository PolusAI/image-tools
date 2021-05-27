# Color Pyramid Builder

This WIPP plugin builds DeepZoom color image pyramids for the Web Deep Zoom Toolkit ([WDZT](https://github.com/usnistgov/WebDeepZoomToolkit)). The [precomputed slide plugin](https://github.com/Nicholas-Schaub/polus-plugins/tree/master/polus-precompute-slide-plugin) can create DeepZoom pyramids, but the pyramids are only grayscale. This plugin permits multiple images to be assigned different colors and overlaid on top of each other within the same pyramid. See the Input Details section for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## ToDo

Currently, if an input stitching vector is provided then at most one image will be created using this plugin. This has to do with the inability of the plugin to properly parse the file pattern and separate out variables associated with the stitching vector versus variables that are not. This should be fixed to identify variables associated with the stitching vector. This could be done by adding a new input, but it isn't ideal since the goal of the plugins is to minimize the number of plugins. A preferred option would be to identify images associated with a plugin so that images are not included in multiple pyramids.

## Input Details

### `--filePattern`

The `filePattern` input uses the [filepattern](https://github.com/Nicholas-Schaub/polus-plugins/tree/master/utils/polus-filepattern-util) utility to parse images. In particular, the `c` variable is parsed and used to assign colors to image in the pyramid using the `layout` parameter. All other parameters included will be placed in different frames.

### `--layout`

The `layout` parameter should be a comma separate list of values up to 7 items long. The position of the item in the list indicates the color assigned to, where the list positions are assigned the following colors:
```
red,green,blue,yellow,magenta,cyan,gray
```

The items in the list should correspond to the channel values extracted for the `filePattern`. As an example, assuming the `filePattern = r001_c{ccc}_z000.ome.tif` and there are 6 files with the following file names:
```
r001_c000_z000.ome.tif
r001_c001_z000.ome.tif
r001_c002_z000.ome.tif
r001_c003_z000.ome.tif
r001_c004_z000.ome.tif
r001_c005_z000.ome.tif
```
Then the list of valid values would be `0-5`.

To assign colors, input the channels in list according to the color you want the image assigned to in the image. Continuing with the example, if channel 3 is a brightfield image and channel 5 is a blue nuclear stain, then a color pyramid with brightfield as gray and nuclei as blue would have the following layout (using the color scheme outlined above):
```
--layout 5,,,,,,3
```

### `--bounds`

The `bounds` parameter is optional, but will rescale the image as it's embedded into the pyramid. This is useful when some channels have higher dynamic range than others. The `bounds` parameter follows similar formatting to the `layout` parameter, where the layout should be a comma separate list corresponding to how each channel should be rescaled.

There are three types of values that can be input into the `bounds` parameter:
1. Blank, then no rescaling will take place.
2. A range of integers will directly rescale pixel intensities to the indicated values. For example, 10000-20000 will rescale pixel values so that values <10000 will be black and values >20000 will be the maximum value.
3. A range of floats will calculate percentiles for the images and rescale according to the percentiles. For example, 0.01-0.99 will first calculate the 1% and 99% pixel values and rescale the the image.

Using the example from the `layout` section, if the first channel was the nuclei channel that should be rescaled to the 0.1% percentile pixel intensity to the 99.9% pixel intensity, and the brightfield image shouldn't be rescaled at all, then the following inputs should be used:
```
--layout 5,,,,,,3 --bounds 0.001-0.999,,,,,,
```

If the brightfield should be rescaled between the specific pixel intensities of 10,000 to 20,000 then the following should be used:
```
--layout 5,,,,,,3 --bounds 0.001-0.999,,,,,,10000-20000
```

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--filePattern` | Filename pattern used to separate data | Input | string |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--layout` | Color ordering (e.g. 1,11,,,,5,6) | Input | string |
| `--bounds` | Set bounds (should be float-float, int-int, or blank, e.g. 0.01-0.99,0-16000,,,,,) | Input | string |
| `--outDir` | Output pyramid path. | Output | pyramid |

