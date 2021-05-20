#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input

# Output paths
outDir=/data/output-vector

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --gpus all \
            labshare/polus-label-to-vector-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir}