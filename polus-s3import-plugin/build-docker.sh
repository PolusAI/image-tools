#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/polus-s3import-plugin:${version}