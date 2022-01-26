#!/bin/bash

#!/bin/bash
version=$(<VERSION)
# datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

#Additional args
methods=Elbow
minimumrange=2
maximumrange=10
numofclus=

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run -v /Users/mezukn/Desktop/polus/s3/data:/data \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-k-means-clustering-plugin:${version} \
            --inpdir ${inpDir} \
            --methods ${methods} \
            --minimumrange ${minimumrange} \
            --maximumrange ${maximumrange} \
            --outdir ${outDir} \