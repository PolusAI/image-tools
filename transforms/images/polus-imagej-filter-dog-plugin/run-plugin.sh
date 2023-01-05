#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName=
in1=/data/input
sigma1=
sigma2=
sigmas1=
sigmas2=

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/polus-imagej-filter-dog-plugin:${version} \
            --opName ${opName} \
            --in1 ${in1} \
            --sigma1 ${sigma1} \
            --sigma2 ${sigma2} \
            --sigmas1 ${sigmas1} \
            --sigmas2 ${sigmas2} \
            --out ${out}
            