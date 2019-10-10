#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/basic-flatfield-plugin:${version}