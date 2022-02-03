#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
opName=ConvolveNaiveF
#opName=PadAndConvolveFFTF
#opName=PadAndConvolveFFT
#opName=ConvolveFFTC
inpDir=/data/input
kernel=/data/kernels
outDir=/data/output

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-filter-convolve-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --kernel ${kernel} \
            --outDir ${out}
            