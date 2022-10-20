#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-deconvolve-richardsonlucytv-plugin:${version}