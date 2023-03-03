# Imaris Parser

This WIPP plugin automatically extracts tracking statistics from the metadata of Imaris .ims files and organizes the data into a .csv format. This cleaned format enables quick visualization of statistical features in plotting software tools such as FlowJo and WIPP Plots as well as reduces the time spent on the manual formatting of Imaris files. 

It also outputs summary statistics as an .xlsx file, formatted similar to the Overall.csv file exported by the Imaris application.

## Run the script

1. Add Imaris .ims files to an image collection in WIPP.
2. Build a workflow in WIPP using the created image collection and the Imaris Parser Plugin. 
3. Upon execution of the workflow, the Imaris file is read and needed data is extracted and stored in temporary csv files. Track ID and Object ID data get linked together and also stored in temporary csv files. Last, data within all temporary csv files is linked to produce the final output: a csv file for each channel and an xlsx file for the overall summary statistics. After the code runs completely, a complete message is logged, and the data is stored in both a csv collection and metadata collection in WIPP.

Contact [Melanie Parham](mailto:melanie.parham@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name           | Description                                           | I/O    | Type          |
| -------------- | ----------------------------------------------------- | ------ | ------------- |
| `--inpdir`     | Input image collection to be processed by this plugin | Input  | collection    |
| `--metaoutdir` | Metadata directory that stores overall data           | Output | collection    |
| `--outdir`     | Output collection                                     | Output | csvCollection |

