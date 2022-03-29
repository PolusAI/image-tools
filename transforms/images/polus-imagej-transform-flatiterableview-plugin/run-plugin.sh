#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
opName=DefaultFlatIterableView
inpDir=/data/input

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-transform-flatiterableview-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --outDir ${outDir}
            