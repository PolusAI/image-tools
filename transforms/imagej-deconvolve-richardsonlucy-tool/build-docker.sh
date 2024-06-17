#!/bin/bash

mkdir segmentation
cp -r ../../segmentation/imagej-threshold-apply-tool ./segmentation/.

version=$(<VERSION)
docker build . -t polusai/imagej-deconvolve-richardsonlucy-tool:${version}

rm -rf ./segmentation
