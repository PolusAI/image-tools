#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-sobel-plugin:${version}