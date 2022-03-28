#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-imagej-transform-flatiterableview-plugin:${version}