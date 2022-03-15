#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-otsu-plugin:${version}