#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
gtDir=/data/groundtruth
predDir=/data/predicted_images
inputClasses=1
filePattern=".+"
fileExtension=".arrow"

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/pixel-segmentation-comparison-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/pixel-segmentation-comparison-plugin:${version} \
            --gtDir ${gtDir} \
            --predDir ${predDir} \
            --inputClasses ${inputClasses} \
            --filePattern ${filePattern} \
            --individualStats \
            --totalStats \
            --fileExtension ${fileExtension} \
            --outDir ${outDir}
