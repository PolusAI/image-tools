#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-yen-plugin:${version}