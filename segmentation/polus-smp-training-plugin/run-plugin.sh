#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../data/smp_training)

# Inputs
pretrainedModel=/data/pretrained_model
modelName="Linknet"
encoderBaseVariantWeights="ResNet,resnet18,imagenet"
#encoderBase="ResNet"
#encoderVariant="resnet34"
#encoderWeights="imagenet"
optimizerName="Adam"
batchSize=8

imagesDir=/data/images
imagesPattern="r{r+}_z{z+}_c{c+}.ome.tif"
labelsDir=/data/labels
labelsPattern="z{z+}.ome.tif"
trainFraction=0.7
segmentationMode="multilabel"

lossName="JaccardLoss"
metricName="IoU"
maxEpochs=100
patience=4
minDelta=1e-4

# Output paths
outputDir=/data/output

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            --rm \
	    --gpus all \
	    --privileged -v /dev:/dev \
            labshare/polus-smp-training-plugin:"${version}" \
            --modelName ${modelName} \
            --encoderBaseVariantWeights ${encoderBaseVariantWeights} \
            --optimizerName ${optimizerName} \
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
