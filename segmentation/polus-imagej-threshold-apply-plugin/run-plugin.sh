#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
opName=ApplyManualThreshold
inpDir=/data/input
threshold=30000
outDir=/data/output

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-threshold-apply-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --threshold ${threshold} \
            --outDir ${outDir}
            