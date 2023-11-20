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

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/region-segmentation-eval:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/region-segmentation-eval:${version} \
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
            --outDir ${outDir}
