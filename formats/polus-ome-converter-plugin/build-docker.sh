#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-ome-converter-plugin:${version}