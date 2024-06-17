#!/bin/bash

cp -r ../imagej-threshold-apply-tool .

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-ij1-tool:${version}

rm -rf imagej-threshold-apply-tool
