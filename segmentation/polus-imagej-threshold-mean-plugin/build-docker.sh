#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-mean-plugin:${version}