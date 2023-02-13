#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/path_to_images
outDir=/data/path_to_output
pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
chunkSize=50
groupBy='c'



# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/filepattern-generator-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir} \
            --pattern ${pattern} \
            --chunkSize ${chunkSize} \
            --groupBy ${groupBy} \
            