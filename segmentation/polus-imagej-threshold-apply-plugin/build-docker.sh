#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-threshold-apply-plugin:${version}