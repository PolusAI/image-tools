# Vector to Label (v0.7.0-dev34)

Here we provide the vector-to-label plugin to convert vector fields to labeled images.
We have new algorithms for this conversion that are more accurate than the ones used in the [Cellpose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) [(github)](https://github.com/MouseLand/cellpose).

This plugin is part of the [WIPP](https://isg.nist.gov/deepzoomweb/software/wipp) ecosystem.
It works with 2d and 3d images of arbitrarily large sizes.

Contact [Nick Schaub](mailto:nick.schaub@nih.gov) or [Najib Ishaq](mailto:najib.ishaq@nih.gov) for more information.

## Algorithm

The vector-to-label plugin uses the concept of flow-fields as described in the [Cellpose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) paper.
Our underlying algorithm, however, is new and will soon be published under "RheaPose".

<!-- The vector-to-label plugin uses the following steps to convert a vector field to a labeled image:

1. Find the geometric-median of each cell in the image.
2. Add heat to each cell's geometric-median.
3. Diffuse the heat outward from each cell's geometric-median.
4. Allow the heat to escape from cell to background, holding the background as a perfect heat sink.
5. Loop from step 2 until the heat has reached a steady state.
6. Every few iterations, add a heat shock to every pixel that has some heat. -->

## Usage

This plugin takes four input parameters:

1. `inpDir`: Input image image collection ("_flow.ome.zarr" files).
2. `filePattern`: Image-name pattern to use when selecting images to process.
3. `flowMagnitudeThreshold`: The minimum flow magnitude at a pixel for it to be considered part of a cell. Defaults to 0.1.
4. `outDir`: Output collection.

## TODO

1. After publishing RheaPose, add link to the paper.

## Building

Run `./build-docker.sh` to build the docker image.

## Install WIPP Plugins

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

The vector-to-label plugin takes three input parameters and one output parameter:

| Name                       | Description                                                                  | I/O    | Type       | Default |
| -------------------------- | ---------------------------------------------------------------------------- | ------ | ---------- | ------- |
| `--inpDir`                 | Input image image collection                                                 | Input  | collection | N/A     |
| `--filePattern`            | Image-name pattern to use when selecting images to process                   | Input  | string     | ".*"    |
| `--flowMagnitudeThreshold` | The minimum flow magnitude at a pixel for it to be considered part of a cell | Input  | number     | 0.1     |
| `--outDir`                 | Output collection                                                            | Output | collection | N/A     |
