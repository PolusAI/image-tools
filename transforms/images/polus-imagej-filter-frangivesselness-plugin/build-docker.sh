#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-frangivesselness-plugin:${version}