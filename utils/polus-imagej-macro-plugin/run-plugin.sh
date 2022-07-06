#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
inpDir=/data/input
macro=/data/macro-blur.txt
maxIterations=15

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-macro-plugin:${version} \
            --inpDir ${inpDir} \
            --macro ${macro} \
            --outDir ${outDir} \
            --maxIterations ${maxIterations}
            