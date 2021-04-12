# Vector to label plugin
Plugin is based off of mask recovery implemented in [Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1). 
The plugin takes a vector field as input and generates masks based on the flow error and cell probability threshold entered by user.

* A meshgrid is generated based on pixel location and pixels are grouped based on where they converge. 
These grouped pixels form a mask . 
* From the masks flows are recomputed and masks above the flow error threshold are removed.
  
* The author's recommended values for cell probability threshold ,flow error and stitching threshold are 0,0.4 and 0. 

* Code used in the plugin is sourced from authors [repository](https://github.com/MouseLand/cellpose/tree/master/cellpose).

* Plugin has been tested on bfio:2.0.4.


Excerpt from paper
  
`We run a dynamical system starting at that pixel location and following the spatial derivatives specified by the horizontal and vertical gradient maps.
We use finite differences with a step size of 1. Note that we do not re-normalize the predicted gradients, but the gradients in the training set have unit norm, so we expect the predicted gradients to be on the same scale. 
We run 200 iterations for each pixel, and at every iteration we take a step in the direction of the gradient at the nearest grid location.
Following convergence, pixels can be easily clustered according to the pixel they end up at. For robustness, we also extend the clusters along regions of high-density of pixel convergence.`


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## To run container locally
  `docker run   -v {zarr file location}:/opt/executables/input -v {output Dir}:/opt/executables/output labshare/polus-vector-label-plugin:{version}  --inpDir /opt/executables/input --outDir /opt/executables/output` 

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 4 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   | Default values    |   
|---------------|-------------------------|--------|--------|--------|
| `--flowThreshold` | flow threshold(threshold  you would want to  have between the  vector recomputed from generated labels and input vector )| Input | number | 0.8   | 
| `--stitchThreshold` | stitch threshold(intersection of union between 2 adjacent slices in a  3d image) | Input | number |   0 | 
| `--inpDir` | Input image collection to be processed by this plugin | Input | GenericData | n/a  | 
| `--cellprobThreshold` | Cell probability threshold(all pixels with prob above threshold are considered for mask identification) | Input | number |   0 | 
| `--outDir` | Output collection | Output | collection | n/a  | 

