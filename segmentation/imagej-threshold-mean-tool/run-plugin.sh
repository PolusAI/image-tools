#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
inpDir=/data/input
pattern=".*"
outDir=/data/output

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-threshold-mean-tool:${version} \
            --inpDir ${inpDir} \
            --pattern ${pattern} \
            --outDir ${outDir}
