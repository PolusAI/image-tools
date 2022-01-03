#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-convolve-plugin:${version}