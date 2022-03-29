#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-transform-flatiterableview-plugin:${version}