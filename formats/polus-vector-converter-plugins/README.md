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
If no `filePattern` is provided, all images in the collection are processed.
Most input parameters use, as defaults, the values recommended by the authors of Cellpose.
The only ones that do not use the authors' recommend default are `flowThreshold` and `maskSizeThreshold`.

`flowMagnitudeThreshold` is used to help locate cell boundaries in an image.
The label-to-vector plugin will produce flows whose magnitudes are 1 for pixels inside cells and 0 for pixels that form the background.
The neural networks used in the cellpose-inference plugin are a bit more noisy.
They will produce flow vectors with:

- high magnitudes, approx. 1, for pixels inside and near the boundaries of cells, and
- low magnitudes, approx 0.005, for pixels in the background or inside cells but far from any cell boundary.

The recommended default is `0.1`.

## TODO

1. The new algorithm for mask reconstruction only works with 2d images for now.I need to work out the math for 3d images; then I'll extend it.
2. Prove the choice to directly use cellpose's algorithms for both plugins.

## Building

We provide one Docker image for each plugin.
The two images are identical except for the entry point.
Run `./build-docker.sh` to build both docker images.

## Install WIPP Plugins

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of either `label_to_vector_plugin.json` or `vector_to_label_plugin.json` (whichever you need to use) into the pop-up window and submit.

## Options

The `label-to-vector` plugin takes 2 input arguments and 1 output argument:

| Name            | Description                                                 | I/O    | Type       |
| --------------- | ----------------------------------------------------------- | ------ | ---------- |
| `--inpDir`      | Input image collection to be processed by this plugin.      | Input  | collection |
| `--filePattern` | Image-name pattern to use when selecting images to process. | Input  | string     |
| `--outDir`      | Output collection.                                          | Output | collection |

The `vector-to-label` plugin takes 3 input arguments and 1 output argument:

| Name                       | Description                                                                   | I/O    | Type       | Defaults |
| -------------------------- | ----------------------------------------------------------------------------- | ------ | ---------- | -------- |
| `--inpDir`                 | Input image collection to be processed by this plugin.                        | Input  | collection | N/A      |
| `--filePattern`            | Image-name pattern to use when selecting images to process.                   | Input  | string     | ".+"     |
| `--flowMagnitudeThreshold` | The minimum flow magnitude at a pixel for it to be considered part of a cell. | Input  | number     | 0.1      |
| `--outDir`                 | Output collection.                                                            | Output | collection | N/A      |
