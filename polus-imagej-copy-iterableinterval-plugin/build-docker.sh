#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagej-copy-iterableinterval-plugin:${version}