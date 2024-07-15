# Label To Vector (v0.7.1)

Here we provide the label-to-vector plugin to convert labeled images to vector fields.
We have new algorithms for this conversion that are more accurate than the ones used in the [Cellpose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) [(github)](https://github.com/MouseLand/cellpose).

This plugin is part of the [WIPP](https://isg.nist.gov/deepzoomweb/software/wipp) ecosystem.
It works with 2d and 3d images of arbitrarily large sizes.

Contact [Nick Schaub](mailto:nick.schaub@nih.gov) or [Najib Ishaq](mailto:najib.ishaq@nih.gov) for more information.

## Algorithm

The label-to-vector plugin uses the concept of flow-fields as described in the [Cellpose](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D) paper.
Our underlying algorithm, however, is new and will soon be published under "RheaPose".

The label-to-vector plugin uses the following steps to convert a labeled image to a vector field:

1. Find the geometric-median of each cell in the image.
   - The geometric-median is the pixel that minimizes the sum of the distances to all other pixels in the cell.
   - We will overload the term "center" to refer to the geometric-median of a cell.
2. Loop:
   1. Add heat to each cell's center.
   2. Diffuse the heat outward from each cell's center.
      - The heat diffusion is done using a convolution with a Gaussian kernel.
   3. Allow the heat to escape from cell to background, holding the background as a perfect heat sink.
   4. If the heat has reached a steady state, stop the loop.
      - The heat has reached a steady state if the heat in every pixel is within some small threshold of the heat in the previous iteration.
   5. Otherwise, every 10 iterations, add a heat shock to every pixel that has some heat.
      - This is done to prevent the heat from getting stuck thinly connected regions.
3. After the loop, the heat values in each cell form a "heat map".
   - Heat will be high in the center of each cell and low at the edges of each cell.
4. The vector field is the vector-gradient of the heat map.

## Usage

This plugin takes three input parameters:

1. `--inpDir`: The input image collection (".ome.tif" or ".ome.zarr" format).
2. `--filePattern`: The image-name pattern to use when selecting images to process. If no file pattern is provided, all images in the collection are processed.
3. `--outDir`: The output image collection (".ome.zarr" format).

## TODO

1. After publishing RheaPose, add link to the paper.

## Building

Run `./build-docker.sh` to build the docker image.

## Install WIPP Plugins

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

The `label-to-vector` plugin takes 2 input arguments and 1 output argument:

| Name            | Description                                                 | I/O    | Type       |
| --------------- | ----------------------------------------------------------- | ------ | ---------- |
| `--inpDir`      | Input image collection to be processed by this plugin.      | Input  | collection |
| `--filePattern` | Image-name pattern to use when selecting images to process. | Input  | string     |
| `--outDir`      | Output collection.                                          | Output | collection |
