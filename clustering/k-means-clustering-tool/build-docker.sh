#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/k-means-clustering-tool:${version}
