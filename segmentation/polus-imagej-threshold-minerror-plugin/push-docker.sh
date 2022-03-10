#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-minerror-plugin:${version}