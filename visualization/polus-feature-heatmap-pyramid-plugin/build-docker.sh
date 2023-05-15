#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feature-heatmap-pyramid-plugin:${version}