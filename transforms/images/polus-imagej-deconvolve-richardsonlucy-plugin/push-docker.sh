#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-deconvolve-richardsonlucy-plugin:${version}