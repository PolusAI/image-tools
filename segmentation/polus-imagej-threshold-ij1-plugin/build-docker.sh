#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-ij1-plugin:${version}