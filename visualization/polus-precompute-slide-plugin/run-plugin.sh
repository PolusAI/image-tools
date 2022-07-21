#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)
echo ${datapath}

# Inputs
inpDir=/data/input
pyramidType=Zarr
imageType=image
filePattern="p02_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}.ome.tif"

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/polus-precompute-slide-plugin:${version} \
            --inpDir ${inpDir} \
            --pyramidType ${pyramidType} \
            --filePattern ${filePattern} \
            --imageType ${imageType} \
            --outDir ${outDir}