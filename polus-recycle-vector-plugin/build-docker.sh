#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/polus-recycle-vector-plugin:${version}