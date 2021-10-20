#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
#opName='GaussRAISingleSigma'
opName='DefaultGaussRAI'
inpDir=/data/input
#sigma=1
sigmas='1,2'

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-filter-gauss-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --sigmas ${sigmas} \
            --outDir ${outDir}
            #--sigma ${sigma} \
            