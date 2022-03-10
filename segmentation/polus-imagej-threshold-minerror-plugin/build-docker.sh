#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-threshold-minerror-plugin:${version}