#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-minerror-plugin:${version}