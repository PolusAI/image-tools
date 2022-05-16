#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-deconvolve-richardsonlucycorrection-plugin:${version}