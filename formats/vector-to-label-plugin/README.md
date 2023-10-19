# Vector to Label (v0.7.0-dev34)

Here we provide the vector-to-label plugin to convert vector fields to labeled images.
We have new algorithms for this conversion that are more accurate than the ones used in the [CellPose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) [(github)](https://github.com/MouseLand/cellpose).

This plugin is part of the [WIPP](https://isg.nist.gov/deepzoomweb/software/wipp) ecosystem.
It works with 2d and 3d images of arbitrarily large sizes.

Contact [Nick Schaub](mailto:nick.schaub@nih.gov) or [Najib Ishaq](mailto:najib.ishaq@nih.gov) for more information.

## Algorithm

The vector-to-label plugin is related to the concept of flow-fields as described in the [CellPose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) paper.
Our underlying algorithm, however, is new and will soon be published under "RheaPose".

## Usage

This plugin takes three parameters:

1. `inpDir`: Input vector-field collection ("_flow.ome.zarr" files).
2. `filePattern`: Image-name pattern to use when selecting images to process.
3. `outDir`: Output collection.

## TODO

1. After publishing RheaPose, add link to the paper.

## Building

Run `./build-docker.sh` to build the docker image.

## Install WIPP Plugins

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

The vector-to-label plugin takes two input parameters and one output parameter:

| Name                       | Description                                                                  | I/O    | Type       | Default |
| -------------------------- | ---------------------------------------------------------------------------- | ------ | ---------- | ------- |
| `--inpDir`                 | Input image image collection                                                 | Input  | collection | N/A     |
| `--filePattern`            | Image-name pattern to use when selecting images to process                   | Input  | string     | ".*"    |
| `--outDir`                 | Output collection                                                            | Output | collection | N/A     |
