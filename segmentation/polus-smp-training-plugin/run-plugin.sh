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

imagesDir=/data/input/train/intensity
imagesPattern="p0_y1_r{r}_c0.ome.tif"
labelsDir=/data/input/train/labels
labelsPattern="p0_y1_r{r}_c0.ome.tif"
trainFraction=0.7
segmentationMode="multilabel"

lossName="MCCLoss"
metricName="IoU"
maxEpochs=5
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
            --imagesDir ${imagesDir} \
            --imagesPattern ${imagesPattern} \
            --labelsDir ${labelsDir} \
            --labelsPattern ${labelsPattern} \
            --trainFraction ${trainFraction} \
            --segmentationMode ${segmentationMode} \
            --lossName ${lossName} \
            --metricName ${metricName} \
            --maxEpochs ${maxEpochs} \
            --patience ${patience} \
            --minDelta ${minDelta} \
            --outputDir ${outputDir}
