#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-maxentropy-plugin:${version}