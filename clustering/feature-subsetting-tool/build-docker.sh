#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feature-subsetting-tool:${version}
