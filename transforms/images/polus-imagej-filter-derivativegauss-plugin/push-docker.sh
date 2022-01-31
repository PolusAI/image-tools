#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-derivativegauss-plugin:${version}