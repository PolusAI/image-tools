#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-rosin-plugin:${version}