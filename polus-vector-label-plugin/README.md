# Vector-label 
  Plugin is based of  mask recovery implemented  in [Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1). 
  
  Excerpts from paper
 
  `We run a dynamical system starting at that pixel location and following the spatial derivatives specified by the horizontal and vertical gradient maps.
   We use finite differences with a step size of 1. Note that we do not re-normalize the predicted gradients, but the gradients in the training set have unit norm, so we expect the predicted gradients to be on the same scale. 
   We run 200 iterations for each pixel, and at every iteration we take a step in the direction of the gradient at the nearest grid location.
   Following convergence, pixels can be easily clustered according to the pixel they end up at. For robustness, we also extend the clusters along regions of high-density of pixel convergence.`

   Code used in the plugin is sourced from authors [repository](https://github.com/MouseLand/cellpose/tree/master/cellpose).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Running container locally
  `docker run   -v {zarr file location}:/opt/executables/input -v {output Dir}:/opt/executables/output labshare/polus-vector-label-plugin:{version}  --inpDir /opt/executables/input --outDir /opt/executables/output` 

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 3 input argument and 1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--flow_threshold` | flow threshold | Input | number |
| `--stitch_threshold` | stitch threshold | Input | number |
| `--inpDir` | Input image collection to be processed by this plugin | Input | GenericData |
| `--cellprob_threshold` | Cell probablity threshold | Input | number |
| `--outDir` | Output collection | Output | collection |

