#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/polus-tiledtiff-converter-plugin:${version}
