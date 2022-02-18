#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-apply-plugin:${version}