#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-montage-plugin:${version}