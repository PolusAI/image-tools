#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../data/smp-training)

# Inputs
#pretrainedModel=/data/pretrained-model
modelName="Linknet"
encoderBase="ResNet"
encoderVariant="resnet34"
encoderWeights="imagenet"
optimizerName="Adam"
batchSize=8

imagesTrainDir=/data/input/train/intensity
labelsTrainDir=/data/input/train/labels
trainPattern="p0_y1_r{r}_c0.ome.tif"

imagesValidDir=/data/input/val/intensity
labelsValidDir=/data/input/val/labels
validPattern="p0_y1_r{r}_c0.ome.tif"

device='cuda'
checkpointFrequency=1

lossName="MCCLoss"
#lossName="DiceLoss"
#lossName="SoftBCEWithLogitsLoss"
maxEpochs=8
patience=2
minDelta=1e-4

# Output paths
outputDir=/data/output

#            --pretrainedModel ${pretrainedModel} \

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            --rm \
            --gpus "all" \
            --privileged -v /dev:/dev \
            labshare/polus-smp-training-plugin:"${version}" \
            --modelName ${modelName} \
            --encoderBase ${encoderBase} \
            --encoderVariant ${encoderVariant} \
            --encoderWeights ${encoderWeights} \
            --optimizerName ${optimizerName} \
            --batchSize ${batchSize} \
            --imagesTrainDir ${imagesTrainDir} \
            --labelsTrainDir ${labelsTrainDir} \
            --trainPattern ${trainPattern} \
            --imagesValidDir ${imagesValidDir} \
            --labelsValidDir ${labelsValidDir} \
            --validPattern ${validPattern} \
            --device ${device} \
            --checkpointFrequency ${checkpointFrequency} \
            --lossName ${lossName} \
            --maxEpochs ${maxEpochs} \
            --patience ${patience} \
            --minDelta ${minDelta} \
            --outputDir ${outputDir}
