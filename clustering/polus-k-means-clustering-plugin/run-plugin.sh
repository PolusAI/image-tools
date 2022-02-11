#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath='/Users/mezukn/Desktop/polus/s3/data/'

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

docker run --mount type=bind,source=${datapath},target=/data/ \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-k-means-clustering-plugin:${version} \
            --inpdir ${inpDir} \
            --methods ${methods} \
            --minimumrange ${minimumrange} \
            --maximumrange ${maximumrange} \
            --outdir ${outDir} \