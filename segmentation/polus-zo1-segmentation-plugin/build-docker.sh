#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-zo1-segmentation-plugin:${version}