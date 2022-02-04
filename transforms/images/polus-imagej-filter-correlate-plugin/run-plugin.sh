#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName=
out_input=/data/input
in1=/data/input
in2=/data/input

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/polus-imagej-filter-correlate-plugin:${version} \
            --opName ${opName} \
            --out_input ${out_input} \
            --in1 ${in1} \
            --in2 ${in2} \
            --out ${out}
            