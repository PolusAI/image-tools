#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName=
in1=/data/input
in2=/data/input

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/polus-imagej-filter-convolve-plugin:${version} \
            --opName ${opName} \
            --in1 ${in1} \
            --in2 ${in2} \
            --out ${out}
            