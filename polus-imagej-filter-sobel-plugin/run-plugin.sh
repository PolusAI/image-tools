#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
#echo $(datapath)

# Inputs
opName="SobelRAI"
inpDir=/data/input

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-sobel-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --outDir ${outDir}
            