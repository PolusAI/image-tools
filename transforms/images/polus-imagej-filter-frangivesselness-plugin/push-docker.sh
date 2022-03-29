#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-frangivesselness-plugin:${version}