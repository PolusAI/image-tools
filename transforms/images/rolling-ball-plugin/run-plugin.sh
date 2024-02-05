#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inputDir=/data/input
ballRadius=25
lightBackground=false

# Output paths
outputDir=/data/output

docker run --mount type=bind,source="${datapath}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            polusai/rolling-ball-plugin:"${version}" \
            --inputDir ${inputDir} \
            --ballRadius ${ballRadius} \
            --lightBackground ${lightBackground} \
            --outputDir ${outputDir}
