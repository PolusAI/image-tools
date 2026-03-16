#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
inpDir=${datapath}/input/images
tabularDir=${datapath}/input/tabular
filePattern="x{x+}_y{y+}_p{p+}_c{c+}.ome.tif"
imageFeature="intensity_image"
padding=0
groupVar="p,c"
percentile=0.8
removeDirection="Below"
writeOutput=true
outDir=${datapath}/output

docker run -v ${datapath}:${datapath} \
            polusai/feature-subsetting-tool:${version} \
            --inpDir ${inpDir} \
            --tabularDir ${tabularDir} \
            --filePattern ${filePattern} \
            --imageFeature${imageFeature} \
            --tabularFeature ${tabularFeature} \
            --padding ${padding} \
            --groupVar ${groupVar} \
            --percentile ${percentile} \
            --groupVar ${groupVar} \
            --removeDirection ${removeDirection} \
            --writeOutput \
            --outDir ${outDir}
