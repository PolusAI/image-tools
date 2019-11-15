#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/stack-z-slices:${version}