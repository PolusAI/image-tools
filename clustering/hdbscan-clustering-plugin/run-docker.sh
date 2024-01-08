#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize .)
echo ${datapath}

# Inputs
inpDir=/data/input
filePattern=".*.csv"
groupingPattern="\w+$"
labelCol=file
averageGroups=true
minClusterSize=10
incrementOutlierId=true
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
           --gpus=all \
           polusai/hdbscan-clustering-plugin:${version} \
           --inpDir ${inpDir} \
           --filePattern ${filePattern} \
	       --groupingPattern ${groupingPattern} \
	       --labelCol ${labelCol} \
           --averageGroups ${averageGroups} \
           --minClusterSize ${minClusterSize} \
           --incrementOutlierId ${incrementOutlierId} \
           --outDir ${outDir}
