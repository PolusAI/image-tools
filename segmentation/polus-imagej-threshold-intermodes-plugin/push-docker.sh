#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-intermodes-plugin:${version}