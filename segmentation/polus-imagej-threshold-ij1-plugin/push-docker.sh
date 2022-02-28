#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-ij1-plugin:${version}