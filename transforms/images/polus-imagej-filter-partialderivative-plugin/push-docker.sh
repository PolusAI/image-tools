#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-partialderivative-plugin:${version}