#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-deconvolve-richardsonlucy-plugin:${version}