# pyramid_generator_3d (0.1.1-dev0)

Generate 3D Image Pyramid from an image collection or Zarr directory. This plugin is a wrapper for argolid.
This tool offers 2 subcommands: `Vol` and `Py3D`, for volume generation and 3D pyramid generation respectively. See [Usage](##usage) section for details.

## Options
| Name        | Description                                                                 | I/O | Type   |
|-------------|-----------------------------------------------------------------------------|-----|--------|
|`--subCmd`   | Subcommand to invoke. Options are `Vol` and `Py3D`.                         |Input|string  |
|`--zarrDir`  | Directory to Zarr arrays for generating 3D pyramid.                         |Input|collection|
|`--inpDir`   | Directory to input image collection. Required if `--zarrDir` is unspecified.|Input|collection|
|`--filePattern` | File pattern for discovering images in `--inpDir`.                       |Input|collection|
|`--groupBy` | Grouping variable for images. Options are `t`, `z`, `c`.                     |Input|string|
|`--outDir`  | Path of output directory.                                                    |Output|collection|
|`--outImgName` | Output name for Zarr arrays when using volume generation.                 |Input|string|
|`--baseScaleKey`| Base scale key for 3D pyramid generation. Default to 0.                  |Input|integer|
|`--numLevels` | Number of levels for 3D pyramid.                                           |Input|integer|

## Usage
### Volume Generation
Use `Vol` subcommand to generate Zarr arrays from image stacks. It reads images from the input directory, groups them by specific dimension, and writes Zarr array into the output directory.
The ***required*** options for `Vol` subcommand are `--inpDir`, `--filePattern`, `--groupBy`, `outDir`, `--outImgName`
Example usage:
```
python3 -m polus.images.formats.pyramid_generator_3d --subCmd Vol --inpDir /path/to/input/images --filePattern img_r{r:ddd}_c{c:ddd}.ome.tif --groupBy c --outDir /path/to/output --outImgName output_image
```

### 3D Pyramid
Use `Py3D` subcommand to generate 3D pyramid from either <ins>(1) a directory with Zarr array</ins> or <ins>(2) a directory of images</ins>.
#### From Zarr directory
When generating from a Zarr directory, the ***required*** options are `--zarrDir`, `--outDir`, and `--numLevels`. `--baseScaleKey` defaults to 0. Since the output will be written into the Zarr directory, use the same directory for `--zarrDir` and `--outDir`.
Example usage:
```
python -m polus.images.formats.pyramid_generator_3d --subCmd Py3D --zarrDir /path/to/zarr/array --outDir /path/to/zarr/array --baseScaleKey 0 --numLevels 2
```

#### From image collection
When generating directly from an image collection, the current tool firsts calls the volume generation routine first to generate Zarr array, from which 3D pyramid is subsequently generated. Thus, all options required for `Vol` subcommand are required in addition to the required options of `Py3D` (excluding `--zarrDir`).
Together, the ***required*** options are `--inpDir`, `--filePattern`, `--groupBy`, `--outDir`, `--outImgName`, `--numLevels`. `--baseScaleKey` defaults to 0.
Example usage:
```
python -m polus.images.formats.pyramid_generator_3d --subCmd Py3D --inpDir /path/to/input/images --filePattern img_r{r:ddd}_c{c:ddd}.ome.tif --groupBy c --outDir /path/to/output --outImgName test_output --baseScaleKey 0 --numLevels 2
```

## Building

To build the Docker image for the tool, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.
