#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-moments-plugin:${version}