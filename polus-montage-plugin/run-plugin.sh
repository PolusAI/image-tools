#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
filePattern=test_p{p+}-z{z+}.ome.tif
inpDir=/data/input
layout=pz

# Optional Inputs
imageSpacing=100
gridSpacing=10

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/polus-montage-plugin:${version} \
            --filePattern ${filePattern} \
            --inpDir ${inpDir} \
            --layout ${layout} \
            --outDir ${outDir}