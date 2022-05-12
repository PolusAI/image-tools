#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
opName=
in1=/data/input
dimension=

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/polus-imagej-filter-partialderivative-plugin:${version} \
            --opName ${opName} \
            --in1 ${in1} \
            --dimension ${dimension} \
            --out ${out}
            