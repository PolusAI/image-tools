#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inpDir=/data/intensity
tileSize=512
modelPath="path/to/model"
filePatternTest="y{y:d+}_r{r:d+}_c0.ome.tif"
filePatternWholeCell = "y{y:d+}_r{r:d+}_c1.ome.tif"
model="mesmerNuclear"
fileExtension=".ome.tif"

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/mesmer-inference-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/mesmer-inference-plugin:${version} \
            --inpDir ${inpDir} \
            --tileSize ${tileSize} \
            --modelPath ${modelPath} \
            --filePatternTest ${filePatternTest} \
            --filePatternWholeCell ${filePatternWholeCell} \
            --model ${model} \
            --fileExtension ${fileExtension} \
            --outDir ${outDir}
