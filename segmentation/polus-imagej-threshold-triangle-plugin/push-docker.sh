#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-triangle-plugin:${version}