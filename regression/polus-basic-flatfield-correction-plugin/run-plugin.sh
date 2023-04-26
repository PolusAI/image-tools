#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)
echo ${datapath}

# Inputs
inpDir=/data/standard/intensity
filePattern="p001_x{x+}_y{y+}_wx{r+}_wy{z+}_c{c}.ome.tif"
darkfield=true
photobleach=false
groupBy="xyrz"

# Output paths
outDir=/data/basic

docker run --mount type=bind,source=${datapath},target=/data/ \
            --gpus=all \
            --user $(id -u):$(id -g) \
            polusai/basic-flatfield-correction-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --darkfield ${darkfield} \
            --photobleach ${photobleach} \
            --groupBy ${groupBy} \
            --outDir ${outDir}
