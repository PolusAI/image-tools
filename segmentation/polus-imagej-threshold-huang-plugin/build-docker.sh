#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-threshold-huang-plugin:${version}