#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-stitching-vector-merger-plugin:${version}