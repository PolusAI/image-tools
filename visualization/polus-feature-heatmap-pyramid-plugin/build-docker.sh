#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-feature-heatmap-pyramid-plugin:${version}