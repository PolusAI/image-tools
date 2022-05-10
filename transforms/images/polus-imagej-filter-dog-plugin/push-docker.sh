#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-dog-plugin:${version}