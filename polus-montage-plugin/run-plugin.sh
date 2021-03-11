#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
filePattern=r01c01f\{p+\}p{z+}-ch1sk1fk1fl1.ome.tif
inpDir=/data/input
layout=pz

# Optional Inputs
imageSpacing=100
gridSpacing=10

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-montage-plugin:${version} \
            --filePattern ${filePattern} \
            --inpDir ${inpDir} \
            --layout ${layout} \
            --outDir ${outDir}