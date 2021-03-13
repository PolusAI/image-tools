#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input
pyramidType=Zarr
imageType=image
filePattern=r01c01f01p{z+}-ch1sk1fk1fl1.ome.tif

# Output paths
outDir=/data/polus-render-ui/pyramids/data

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-precompute-slide-plugin:${version} \
            --inpDir ${inpDir} \
            --pyramidType ${pyramidType} \
            --imageType ${imageType} \
            --outDir ${outDir} \
            --filePattern ${filePattern}