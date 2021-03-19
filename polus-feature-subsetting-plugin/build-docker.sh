#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-feature-subsetting-plugin:${version}