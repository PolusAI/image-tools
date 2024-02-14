#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/outlier-removal-tool:${version}
