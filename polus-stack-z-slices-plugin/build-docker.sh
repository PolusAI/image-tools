#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-stack-z-slices-plugin:${version}