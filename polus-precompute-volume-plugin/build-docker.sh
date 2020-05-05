#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-precompute-volume-plugin:${version}