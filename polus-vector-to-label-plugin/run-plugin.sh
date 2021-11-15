#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/output-vector

# Output paths
outDir=/data/output

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --gpus all \
            labshare/polus-vector-label-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir}