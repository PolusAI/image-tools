#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-filter-convolve-plugin:${version}