#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
#opName='DoGSingleSigmas'
opName='DoGVaryingSigmas'
inpDir=/data/input
sigma1=2
sigma2=3
sigmas1=1,2
sigmas2=1,2

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-filter-dog-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --sigmas1 ${sigmas1} \
            --sigmas2 ${sigmas2} \
            --outDir ${outDir}
            