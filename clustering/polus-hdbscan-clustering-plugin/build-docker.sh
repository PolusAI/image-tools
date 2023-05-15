#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/hdbscan-clustering-plugin:${version}