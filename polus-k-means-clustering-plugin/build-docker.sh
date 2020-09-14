#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-k-means-clustering-plugin:${version}
