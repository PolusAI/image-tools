#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-gauss-plugin:${version}