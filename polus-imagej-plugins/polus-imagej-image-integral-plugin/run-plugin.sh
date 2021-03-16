#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName=DefaultIntegralImg
in1=/data/input

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-imagej-image-integral-plugin:${version} \
            --opName ${opName} \
            --in1 ${in1} \
            --out ${out}
            