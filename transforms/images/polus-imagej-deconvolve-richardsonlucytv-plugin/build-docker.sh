#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-deconvolve-richardsonlucytv-plugin:${version}