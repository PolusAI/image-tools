#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
inpDir=${datapath}/input
filePattern=".*.csv"
groupingPattern="\w+$"
labelCol="species"
minClusterSize=3
outDir=${datapath}/output

docker run -v ${datapath}:${datapath} \
            polusai/hdbscan-clustering-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --groupingPattern ${groupingPattern} \
            --labelCol ${labelCol} \
            --minClusterSize ${minClusterSize} \
            --incrementOutlierId \
            --outDir ${outDir}
