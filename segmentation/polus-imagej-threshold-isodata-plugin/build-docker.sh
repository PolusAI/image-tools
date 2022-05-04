#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-isodata-plugin:${version}