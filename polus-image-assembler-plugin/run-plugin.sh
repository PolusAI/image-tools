#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
stitchPath=/data/input_vector
imgPath=/data/input_stitched
timesliceNaming=false
filePattern=".**"

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/polus-image-assembler-plugin:${version} \
            --stitchPath ${stitchPath} \
            --imgPath ${imgPath} \
            --timesliceNaming ${timesliceNaming} \
            --filePattern {filePattern} \
            --outDir ${outDir}
