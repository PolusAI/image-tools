#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input
filePattern=x{xxx}-y{yyy}-z{zzz}.ome.tif
darkfield=true
photobleach=false
groupBy='p'

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-basic-flatfield-correction-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --darkfield ${darkfield} \
            --photobleach ${photobleach} \
            --groupBy ${groupBy} \
            --outDir ${outDir}
