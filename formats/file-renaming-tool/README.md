# File Renaming(v0.2.5)
This WIPP plugin renames files in an image collection (typically microscopy images) using powerful, user-defined filename patterns for both input matching and output naming.

It is particularly useful for:
- Standardizing file names across experiments
- Converting channel names (GFP, DAPI, TXRED…) to numeric indices
- Adding zero-padding consistently
- Reorganizing naming schemes
- Working with nested directory structures

## Example Usage
* The user can upload an image collection where all files contain similar
naming conventions.

 * **Input collection:**
`img_x01_y01_DAPI.tif`
`img_x01_y01_GFP.tif`
`img_x01_y01_TXRED.tif`

 * **Output collection:**
`newdata_x001_y001_c001.tif`
`newdata_x001_y001_c002.tif`
`newdata_x001_y001_c003.tif`

 * **filePattern:**
`img_x{row:dd}_y{col:dd}_{channel:c+}.ome.tif`

 * **outFilePattern:**
`newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.ome.tif`

**Important rules:**

* The user can format the output digit using the number of digits
specified in the output format.
 * `d` represents *digit*
 * `c` represents *character*.

* Note that c+ only matches letters in the alphabet, not symbols and numbers

* If the output formats have plus signs (+), then the number of output
digits/characters is not fixed.

* Finally, the input and output pattern data types *must* agree with one
exception:
 * If the input is a chracter and the output is digit,
then the script sorts the strings that match the character pattern and
assigns numbers 0+ to them.

* Implemented a new optional boolean feature `mapDirectory` to append mapped directory names in renamed files.


## Renaming files within a complex nested directory structure:
In specific scenarios where users need to rename files within nested subdirectories, this functionality can be leveraged by providing an appropriate pattern

For Example

```
BBBC001
    └── raw
        ├── Ground_Truth
        │   └── groundtruth_images
        │       ├── AS_09125_050118150001_A03f00d0.tif
        │       ├── AS_09125_050118150001_A03f01d0.tif
        │       ├── AS_09125_050118150001_A03f02d0.tif
        │       ├── AS_09125_050118150001_A03f03d0.tif
        │       ├── AS_09125_050118150001_A03f04d0.tif
        │       └── AS_09125_050118150001_A03f05d0.tif
        └── Images
            └── human_ht29_colon_cancer_1_images
                ├── AS_09125_050118150001_A03f00d0.tif
                ├── AS_09125_050118150001_A03f01d0.tif
                ├── AS_09125_050118150001_A03f02d0.tif
                ├── AS_09125_050118150001_A03f03d0.tif
                ├── AS_09125_050118150001_A03f04d0.tif
                └── AS_09125_050118150001_A03f05d0.tif

```

Now, renaming files within the `human_ht29_colon_cancer_1_images` is achievable by providing a `filepattern` such as `/.*/Images/(?P<directory>.*)/.*_{row:c}{col:dd}f{f:dd}d{channel:d}.tif`, and specifying `outFilePattern` as `x{row:dd}_y{col:dd}_p{f:dd}_c{channel:d}.tif`. If the mapDirectory option is not utilized, the raw directory name will be appended in the renamed files. To handle directory names containing both letters and digits, employ `(?P<directory>.*)`; use `{directory:c+}` or `{directory:d+}` if it contains solely letters or digits, respectively.

#### Note:
To extract directory names, the pattern should start with a backslash



Contact [Melanie Parham](mailto:melanie.parham@axleinfo.com), [Hamdah Shafqat abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more
information.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Docker / CLI Examples

Basic

```
docker run --rm \
  -v "/path/to/input/images:/data/input" \
  -v "/path/to/output:/data/output" \
  polusai/file-renaming-tool:0.2.5 \
    --inpDir         /data/input \
    --outDir         /data/output \
    --filePattern    'img_x{row:dd}_y{col:dd}_{channel:c+}.tif' \
    --outFilePattern 'r{row:03d}_c{col:03d}_ch{channel:03d}.ome.tif'


```
Directory mapping

```
docker run --rm \
  -v "/path/to/dataset:/data" \
  polusai/file-renaming-tool:0.2.5 \
    --inpDir         /data/input \
    --outDir         /data/output \
    --filePattern    '/.*/Images/(?P<directory>.*)/.*_{row:c}{col:dd}f{f:dd}d{channel:d}.tif' \
    --outFilePattern 'x{row:dd}_y{col:dd}_p{f:dd}_c{channel:d}_dir{directory}.tif' \
    --mapDirectory


```


## Options

This plugin takes three input argument and one output argument:

| Name               | Description                       | I/O      | Type       |
|--------------------|-----------------------------------|----------|------------|
| `--inpDir`         | Input image collection            | Input    | collection |
| `--filePattern`    | Input filename pattern            | Input    | string     |
| `--outDir`         | Output collection                 | Output   | collection |
| `--outFilePattern` | Output filename pattern           | Input    | string     |
| `--mapDirectory`   | Extract mapped directory name     | Input    | boolean    |
| `--preview`        | Generate a JSON file with outputs | Output   | JSON       |
