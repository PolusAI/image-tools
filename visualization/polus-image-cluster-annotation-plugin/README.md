# Image cluster annotation

The image cluster annotation plugin converts the original image as all zeros except at the borders which contains the cluster id. The inputs for this plugin is an image collection and csv collection. The input csv file should have a column with filenames. The filename in input csvfile should match with the image name in image collection. The output for this plugin is an image collection.

## Inputs:
### Input image collection:
The input image file that need to be converted as all zeros except at the borders. This is a required parameter for the plugin.

### Input csv collection:
The input csv file that contains the cluster id. The file should be in csv format. This is a required parameter for the plugin.

### Border width:
Enter the required thickness at the borders that should be changed as cluster id. Default value is 2.

## Output:
The output is a image file collection contains all zeros except at the borders.The border pixels will be cluster id.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes four input argument if methods other than 'Manual' is selected else three input arguments and one output argument:

| Name            | Description            | I/O    | Type            |
| --------------- | ---------------------- | ------ | --------------- |
| `--imgdir`      | Input image collection | Input  | imageCollection |
| `--csvdir`      | Input csv collection   | Input  | csvCollection   |
| `--borderwidth` | Enter border width     | Input  | integer         |
| `--outdir`      | Output collection      | Output | imageCollection |


