#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input_stack
pyramidType=Neuroglancer
imageType=image
# filePattern=

# Output paths
outDir=/data/neuroglancer/dist/dev

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-precompute-slide-plugin:${version} \
            --inpDir ${inpDir} \
            --pyramidType ${pyramidType} \
            --imageType ${imageType} \
            --outDir ${outDir}
            # --filePattern ${filePattern} \