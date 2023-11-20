#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-percentile-plugin:${version}