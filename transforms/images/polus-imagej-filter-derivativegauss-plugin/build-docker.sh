#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-filter-derivativegauss-plugin:${version}