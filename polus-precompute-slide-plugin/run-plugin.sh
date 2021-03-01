#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input
pyramidType=Neuroglancer
imageType=image
filePattern=

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-image-assembler-plugin:${version} \
            --stitchPath ${stitchPath} \
            --imgPath ${imgPath} \
            --outDir ${outDir} \
            --timesliceNaming ${timesliceNaming}