#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
#opName=PadAndCorrelateFFT
opName=CorrelateFFTC
inpDir=/data/input
kernel=/data/kernels

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-filter-correlate-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --kernel ${kernel} \
            --outDir ${outDir}
            