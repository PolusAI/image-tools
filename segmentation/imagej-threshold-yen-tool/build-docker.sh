#!/bin/bash

cp -r ../imagej-threshold-apply-tool .

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-yen-tool:${version}

rm -rf imagej-threshold-apply-tool
