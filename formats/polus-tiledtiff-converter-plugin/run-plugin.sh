#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
input=/data/input

# Output paths
output=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-tiledtiff-converter-plugin:${version} \
            --input ${input} \
            --output ${output}
