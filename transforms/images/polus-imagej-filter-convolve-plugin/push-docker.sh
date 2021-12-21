#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-filter-convolve-plugin:${version}