#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-filter-frangivesselness-plugin:${version}