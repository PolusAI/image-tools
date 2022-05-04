#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/path_to_images
outDir=/data/path_to_output
pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
groupBy='c' 
# or 
# groupBy=None



# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/remove-border-objects-plugin:${version} \
            --inpDir ${inpDir} \
            --pattern ${pattern} \
            --groupBy ${groupBy} \
            --outDir ${outDir} 
            