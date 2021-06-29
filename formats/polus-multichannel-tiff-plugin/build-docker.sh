#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-multichannel-tiff-plugin:${version}