#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-aics-classic-seg-plugin:${version}