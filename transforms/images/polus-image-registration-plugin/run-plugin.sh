#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
inpDir=/data/inputs
filepattern="S1_R{r}_C1-C11_A1_y0(00-14)_x0(00-21)_c0{cc}.ome.tif"
template="S1_R1_C1-C11_A1_y0(00-14)_x0(00-21)_c000.ome.tif"
registrationVariable=r
transformationVariable=c
method=Projective

# Output paths
outDir=/data/outputs

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/polus-image-registration-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filepattern} \
            --template ${template} \
            --registrationVariable ${registrationVariable} \
            --TransformationVariable ${transformationVariable} \
            --method ${method} \
            --outDir ${outDir}
