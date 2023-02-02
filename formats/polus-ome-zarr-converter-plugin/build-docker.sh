#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/ome-zarr-converter-plugin:${version}