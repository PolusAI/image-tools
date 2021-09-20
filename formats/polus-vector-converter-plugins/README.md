# Vector Converter Plugins

Here we provide two plugins to convert to and from flow-field vectors as described by [Cellpose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) [(github)](https://github.com/MouseLand/cellpose).

Both plugins have been designed to scale to 2d or 3d images of arbitrarily large sizes.

Contact [Najib Ishaq](mailto:najib.ishaq@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

### Label to Vector

This WIPP plugin converts a labelled image to vector field and stores the vector field as an ome zarr.
For 2d images, there are 4 channels: `cell_probability`, `flow_y`, `flow_x` and `labels`.
For 3d images, there are 5 channels: `cell_probability`, `flow_z`, `flow_y`, `flow_x` and `labels`.
The 'labels' channel is stored for reference.
If no `filePattern` is provided, all images in the collection are processed.

### Vector to Label

This plugin turns Cellpose flow fields into a labeled image.
It works with either 2d or 3d flow fields.
If no `filePattern` is provided, all images in the collection are processed.
Most input parameters use, as defaults, the values recommended by the authors of Cellpose.
The only ones that do not use the authors' recommend default are `flowThreshold` and `maskSizeThreshold`.

For `flowThreshold`, this is because Cellpose inference will often produce a large number small sized flow-field wells.
These each become a labelled object in an intermediate image.
Cellpose then calculates a flow-field for each such object to compare against the flow-fields given by the user.
Objects corresponding to flow-fields with large errors are discarded.
However, because there are often a very large number of small objects in flow-fields produced by Cellpose inference, this step takes an inordinately long time.
If you feel comfortable spending the extra time, the authors' recommended value is `0.4`.

For `maskSizeThreshold`, the functions for flow-field dynamics in Cellpose were developed under the assumption that they would get to process an entire image at once.
Thus, if some object (recovered from the flow-fields) covered a large enough fraction of the image, the authors set this fraction to `0.4`, that object would be considered part of the background.
With the more scalable implementation provided here, there is a decent chance that some objects will be large enough to cover most, if not all, of a tile.
We would rather not label such objects as background.
We use a tile-size of `1024x1024` for 2d images and `1024x1024x1024` for 3d images.
If you are certain that none of the relevant objects in your images are large enough to cover a significant fraction of a tile, feel free to set `maskSizeThreshold` to a value of your choice.

## TODO

1. For now, interpolation of flow-fields for 3d images is only available on the GPU and not on the CPU.
   Use equations from [this article](https://en.wikipedia.org/wiki/Trilinear_interpolation).
2. Cellpose's method for merging nearby sinks when following flows consumes far too much memory and is the major bottleneck in the code for recovering masks from flow fields.
   It might be better, for memory and time, to use a connected-component analysis of some variant of a neighborhood graph of the sinks.

## Building

We provide one Docker image for each plugin.
The two images are identical except for the entry point.
Run `./build-docker.sh` to build both docker images.

## Install WIPP Plugins

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of either `label_to_vector_plugin.json` or `vector_to_label_plugin.json` (whichever you need to use) into the pop-up window and submit.

## Options

The `label-to-vector` plugin takes 2 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin. | Input | collection |
| `--filePattern` | Image-name pattern to use when selecting images to process. | Input | string |
| `--outDir` | Output collection. | Output | collection |

The `vector-to-label` plugin takes 8 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   | Defaults |
|---------------|-------------------------|--------|--------|----------|
| `--inpDir` | Input image collection to be processed by this plugin. | Input | collection | N/A |
| `--filePattern` | Image-name pattern to use when selecting images to process. | Input | string | ".+" |
| `--flowThreshold` | Flow-error Threshold. Margin between flow-fields computed from labelled masks against input flow-fields. | Input | number | 1.0 |
| `--maskSizeThreshold` | Maximum fraction of a tile that a labelled object can cover. | Input | number | 1.0 |
| `--interpolate` | Whether to use bilinear/trilinear interpolation on 2d/3d flow-fields respectively. | Input | boolean | true |
| `--cellprobThreshold` | Cell Probability Threshold. | Input | number | 0.4 |
| `--numIterations` | Number of iterations for which to follow flows. | Input | number | 200 |
| `--minObjectSize` | Minimum number of pixels for an object to be valid. Object with fewer pixels are removed. | Input | number | 15 |
| `--outDir` | Output collection. | Output | collection | N/A |
