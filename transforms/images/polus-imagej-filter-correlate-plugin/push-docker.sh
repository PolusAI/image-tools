#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-correlate-plugin:${version}