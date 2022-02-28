#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-threshold-ij1-plugin:${version}