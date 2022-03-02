#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-threshold-intermodes-plugin:${version}