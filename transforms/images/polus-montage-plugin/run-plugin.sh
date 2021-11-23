#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
filePattern=S1_R{r}_C1-C11_A1_y0{yy}_x0{xx}_c00{c}.ome.tif
inpDir=/data/images/MaricRatBrain2019/subset/intensity
layout=c,xy,r

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