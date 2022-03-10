#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-minerror-plugin:${version}