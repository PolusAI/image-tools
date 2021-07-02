#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inputDir=/data/input
filePattern="IowaFull_z{z+}"
groupBy="z"
axes="both"
smoothing="true"


# Output paths
outputDir=/data/output

docker run --mount type=bind,source="${datapath}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            labshare/polus-autocropping-plugin:"${version}" \
            --inputDir ${inputDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --axes ${axes} \
            --smoothing ${smoothing} \
            --outputDir ${outputDir}
            