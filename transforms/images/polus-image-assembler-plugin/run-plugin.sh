#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
stitchPath=/data/vector
imgPath=/data/input
timesliceNaming=false

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/polus-image-assembler-plugin:${version} \
            --stitchPath ${stitchPath} \
            --imgPath ${imgPath} \
            --timesliceNaming ${timesliceNaming} \
            --outDir ${outDir}
