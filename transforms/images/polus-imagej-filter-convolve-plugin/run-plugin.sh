#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName='ConvolveNaiveF'
inpDir=/data/input
weights=/data/input

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-filter-convolve-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --weights ${weights} \
            --outDir ${outDir}
            