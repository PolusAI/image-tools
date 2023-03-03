#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/zo1-segmentation-plugin:${version}