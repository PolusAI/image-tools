#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input
connectivity="1"


# Output paths
outDir=/data/output

docker run --mount type=bind,source="${datapath}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            labshare/polus-ftl-label-plugin:"${version}" \
            --inpDir ${inpDir} \
            --connectivity ${connectivity} \
            --outDir ${outDir}