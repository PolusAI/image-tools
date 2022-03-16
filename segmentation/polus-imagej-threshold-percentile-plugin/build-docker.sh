#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-threshold-percentile-plugin:${version}