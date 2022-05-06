#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
#opName=RichardsonLucyC
opName=PadAndRichardsonLucy
inpDir=/data/input
psf=/data/kernels
maxIterations=5

# Output paths
out=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/imagej-deconvolve-richardsonlucy-plugin:${version} \
            --opName ${opName} \
            --inpDir ${inpDir} \
            --psf ${psf} \
            --maxIterations ${maxIterations} \
            --out ${out}
            