#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
opName='DefaultFrangi'
out_input=/data/input
inpDir=/data/input
spacing='1,1'
scale=1

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-filter-frangivesselness-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --spacing ${spacing} \
            --scale ${scale} \
            --outDir ${outDir}
     