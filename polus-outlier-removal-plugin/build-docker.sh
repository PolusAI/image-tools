#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-outlier-removal-plugin:${version}
