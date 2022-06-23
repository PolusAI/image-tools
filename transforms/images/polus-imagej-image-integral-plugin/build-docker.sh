#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-image-integral-plugin:${version}