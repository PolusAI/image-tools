#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-threshold-triangle-plugin:${version}