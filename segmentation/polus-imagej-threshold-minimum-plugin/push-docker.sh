#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-minimum-plugin:${version}