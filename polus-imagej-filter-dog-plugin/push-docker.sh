#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-filter-dog-plugin:${version}