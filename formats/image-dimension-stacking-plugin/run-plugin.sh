#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/path_to_files

## stacking z dimension
filePattern="tubhiswt_C1-z{z:d+}.ome.tif"
# ## stacking c dimension
# filePattern="x{x+}_y{y+}_p01_c{c+}.ome.tif"
# ## stacking t dimension
# filePattern="img00001_t{t:d+}_ch0.ome.tif"
groupBy = "z"
# Output paths
outDir=/data/path_to_output

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/image-dimension-stacking-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --outDir ${outDir}
