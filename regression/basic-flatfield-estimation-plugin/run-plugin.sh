#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ./data)
echo "${datapath}"

docker run polusai/basic-flatfield-estimation-plugin:"${version}"

# Parameters
inpDir=/data/inputs
outDir=/data/outputs
filePattern="S1_R{r:d}_C1-C11_A1_y00{y:d}_x0{x:dd}_c0{c:dd}.ome.tif"
groupBy="cr"

#            --gpus=all \
docker run --mount type=bind,source="${datapath}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            polusai/basic-flatfield-estimation-plugin:"${version}" \
            --inpDir ${inpDir} \
            --outDir ${outDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --getDarkfield
