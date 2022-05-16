#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-deconvolve-richardsonlucycorrection-plugin:${version}