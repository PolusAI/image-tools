#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
stitchDir=/data/vector
collectionDir=/data/images/MaricRatBrain2019/fovs/intensity
filepattern=.+

# Output paths
outDir=/data/output

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-recycle-vector-plugin:${version} \
            --stitchDir ${stitchDir} \
            --collectionDir ${collectionDir} \
            --outDir ${outDir} 
            