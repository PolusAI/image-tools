#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-filter-frangivesselness-plugin:${version}