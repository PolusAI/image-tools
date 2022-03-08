#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-maxlikelihood-plugin:${version}