#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-maxlikelihood-plugin:${version}