#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/polus-file-renaming-plugin:${version}