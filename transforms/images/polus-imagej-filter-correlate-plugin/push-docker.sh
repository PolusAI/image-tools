#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-filter-correlate-plugin:${version}