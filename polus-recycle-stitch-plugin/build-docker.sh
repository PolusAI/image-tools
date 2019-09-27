#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/polus-recycle-stitch-plugin:${version}