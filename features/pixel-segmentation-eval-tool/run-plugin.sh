#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
gtDir=/data/groundtruth
predDir=/data/predicted_images
inputClasses=1
filePattern=".+"

# Output paths
outDir=/data/output

# Show the help options
# docker run polusai/pixel-segmentation-eval:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/pixel-segmentation-eval:${version} \
            --gtDir ${gtDir} \
            --predDir ${predDir} \
            --inputClasses ${inputClasses} \
            --filePattern ${filePattern} \
            --individualStats \
            --totalStats \
            --outDir ${outDir}
