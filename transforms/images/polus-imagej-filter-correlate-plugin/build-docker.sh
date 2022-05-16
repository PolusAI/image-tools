#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-correlate-plugin:${version}