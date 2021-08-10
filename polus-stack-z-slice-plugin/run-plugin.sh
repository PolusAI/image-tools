#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input
filePattern=r01c01f01p{z+}-ch1sk1fk1fl1.ome.tif

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-stack-z-slice-plugin:${version} \
            --filePattern ${filePattern} \
            --inpDir ${inpDir} \
            --outDir ${outDir} 