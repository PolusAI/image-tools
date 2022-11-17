#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/imagej-filter-convolve-plugin:${version}