#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/outlier-removal-plugin:${version}
