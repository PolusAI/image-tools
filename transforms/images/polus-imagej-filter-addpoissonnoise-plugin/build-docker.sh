#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-addpoissonnoise-plugin:${version}