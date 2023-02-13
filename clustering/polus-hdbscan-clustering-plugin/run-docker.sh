#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize .)
echo ${datapath}

# Inputs
inpDir=/data/input
groupingPattern="(\d+)-\d+-\d+"
labelCol=file
averageGroups=true
minClusterSize=10
incrementOutlierId=true
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
           --gpus=all \
           labshare/polus-hdbscan-clustering-plugin:${version} \
           --inpDir ${inpDir} \
	       --groupingPattern ${groupingPattern} \
	       --labelCol ${labelCol} \
           --averageGroups ${averageGroups} \
           --minClusterSize ${minClusterSize} \
           --incrementOutlierId ${incrementOutlierId} \
           --outDir ${outDir}
