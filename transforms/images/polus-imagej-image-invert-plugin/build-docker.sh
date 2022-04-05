#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-image-invert-plugin:${version}