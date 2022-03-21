#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-renyientropy-plugin:${version}