#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
opName='ApplyThresholdMethod$Moments'
inpDir=/data/input

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-threshold-moments-plugin:${version} \
            --opName ${opName} \
            --inpDir ${in1} \
            --outDir ${out}
            