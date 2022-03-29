#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName=
out_input=/data/input
in1=/data/input
spacing=
scale=

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/polus-imagej-filter-frangivesselness-plugin:${version} \
            --opName ${opName} \
            --out_input ${out_input} \
            --in1 ${in1} \
            --spacing ${spacing} \
            --scale ${scale} \
            --out ${out}
            