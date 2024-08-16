# ImageJ deconvolve richardsonlucytv v(0.5.1)

This plugin applies the [Richardson-Lucy Deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution) to input collection with
a total variation regularization factor as described in (Richardson-Lucy
algorithm with total variation regularization for 3D confocal microscope
deconvolution Microsc Res Rech 2006 Apr; 69(4)- 260-6). This is an iterative
process that can recover an underlying blurred image if the psf
(point spread function) mask of the image is known or can be estimated.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

Bump the version in the `VERSION` file.

Then to build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name                     | Description                                     | I/O    | Type       |
| ------------------------ | ----------------------------------------------- | ------ | ---------- |
| `--inpDir`               | Input collection to be processed by this plugin | Input  | collection |
| `--pattern`              | The filepattern to use for the images           | Input  | string     |
| `--psf`                  | The point spread function mask to be used       | Input  | collection |
| `--maxIterations`        | The maximum number of algorithm iterations      | Input  | number     |
| `--regularizationFactor` | The regularization factor to use                | Input  | number     |
| `--outDir`               | The output collection                           | Output | collection |
