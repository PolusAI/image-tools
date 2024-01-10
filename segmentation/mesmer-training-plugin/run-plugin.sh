#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
trainingImages=/data/train_images
trainingLabels=/data/train_labels
testingImages=/data/test_images
testingLabels=/data/test_labels
modelBackbone="resnet50"
filePattern="y{y+}_r{r+}_c0.ome.tif"
tileSize=512
iterations=10
batchSize=30

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/mesmer-training-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/mesmer-training-plugin:${version} \
            --trainingImages ${trainingImages} \
            --trainingLabels ${trainingLabels} \
            --testingImages ${testingImages} \
            --testingLabels ${testingLabels} \
            --modelBackbone ${modelBackbone} \
            --filePattern ${filePattern} \
            --tilesize ${tilesize} \
            --iterations ${iterations} \
            --batchSize ${batchSize} \
            --outDir ${outDir}