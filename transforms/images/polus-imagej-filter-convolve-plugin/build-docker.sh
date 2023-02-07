#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-convolve-plugin:${version}