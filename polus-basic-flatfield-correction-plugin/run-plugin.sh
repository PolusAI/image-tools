#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input
filePattern=r01c01f{p+}p0{z}-ch1sk1fk1fl1.ome.tif
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
