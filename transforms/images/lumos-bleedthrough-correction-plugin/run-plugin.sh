#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ./data)
echo "${datapath}"

docker run polusai/lumos-bleedthrough-correction-plugin:"${version}"

# Parameters
inpDir=/data/inputs
filePattern="S1_R{r:d}_C1-C11_A1_y00{y:d}_x0{x:dd}_c0{c:dd}.ome.tif"
groupBy="cr"
numFluorophores=10
outDir=/data/outputs

docker run --mount type=bind,source="${datapath}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            polusai/lumos-bleedthrough-correction-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --numFluorophores ${numFluorophores} \
            --outDir ${outDir}
