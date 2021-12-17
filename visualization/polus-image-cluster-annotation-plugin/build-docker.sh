#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-image-cluster-annotation-plugin:${version}