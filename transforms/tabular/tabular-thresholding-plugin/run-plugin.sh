#!/bin/bash

# version=$(<VERSION)
version=$(<VERSION)
ln -s /usr/local/bin/greadlink /usr/local/bin/readlink
datapath=$(readlink --canonicalize ../data)

#Inputs
inpDir=/data/inp
metaDir=/data/meta
outDir=/data/out
# mappingvariableName is the featureName which is common between two CSVs and used for merging data
mappingvariableName='intensity_image'
# Name of the variable which has information of untreated cell labels in binary format for example (0,1)
negControl='virus_negative'
# Name of the variable which has information of cell labels with the known treatment also in binary format for example (0,1)
posControl='virus_neutral'
# Name of the variable used for thresholding
variableName='MEAN'
thresholdType='all'
numBins=512
falsePositiverate=0.1
n=4
outFormat="feather"


# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO
docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/tabular-thresholding-plugin:${version} \
            --inpDir ${inpDir} \
            --metaDir ${metaDir} \
            --outDir ${outDir} \
            --mappingvariableName ${mappingvariableName} \
            --negControl ${negControl} \
            --posControl ${posControl} \
            --variableName ${variableName} \
            --thresholdType ${thresholdType} \
            --falsePositiverate ${falsePositiverate} \
            --numBins ${numBins} \
            --n ${n} \
            --outFormat ${outFormat}