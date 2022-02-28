#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-threshold-ij1-plugin:${version}