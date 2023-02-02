#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-deconvolve-richardsonlucy-plugin:${version}