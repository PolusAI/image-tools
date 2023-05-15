#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/stitching-vector-merger-plugin:${version}