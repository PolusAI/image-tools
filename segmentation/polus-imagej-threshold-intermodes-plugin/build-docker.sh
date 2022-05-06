#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-intermodes-plugin:${version}