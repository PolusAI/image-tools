#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-otsu-plugin:${version}