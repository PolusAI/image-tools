#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input

while getopts o: flag
do
    case "${flag}" in
        o) opName=${OPTARG}
    esac
done

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-image-integral-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --outDir ${outDir}
            