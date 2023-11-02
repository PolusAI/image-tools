#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/zo1-border-segmentation-plugin:${version}
