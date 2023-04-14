#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
gtDir=/data/groundtruth
predDir=/data/predicted_images
inputClasses=1
radiusFactor=0.5
iouscore=0.
filePattern=".+"
fileExtension=".arrow"

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/cellular-eval-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/cellular-eval-plugin:${version} \
            --gtDir ${gtDir} \
            --predDir ${predDir} \
            --inputClasses ${inputClasses} \
            --individualData \
            --individualSummary \
            --totalStats \
            --totalSummary \
            --radiusFactor ${radiusFactor} \
            --iouScore ${iouScore} \
            --filePattern ${filePattern} \
            --fileExtension ${fileExtension} \
            --outDir ${outDir}
