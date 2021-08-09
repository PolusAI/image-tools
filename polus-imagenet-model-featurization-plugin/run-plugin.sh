#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/inputs
outDir=/data/outputs
model=VGG19
resolution=500x500

docker run --mount type=bind,source=${datapath},target=/data/ --gpus=all \
            labshare/polus-imagenet-model-featurization-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir} \
            --model ${model} \
            --resolution ${resolution}
