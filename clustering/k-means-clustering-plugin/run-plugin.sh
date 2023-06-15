#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inp_dir=/data/inputs

# Output paths
outDir=/data/output

#Additional args
filePattern=".csv"
methods=Elbow
minimumrange=2
maximumrange=10
numofclus=3

# Show the help options
# docker run polusai/k-means-clustering-plugin:${version}

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/k-means-clustering-plugin:${version} \
            --inpdir ${inpDir} \
            --filePattern ${filePattern} \
            --methods ${methods} \
            --minimumrange ${minimumrange} \
            --maximumrange ${maximumrange} \
            --outdir ${outDir} \
