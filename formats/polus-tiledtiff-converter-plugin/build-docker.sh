#!/bin/bash
version=$(<VERSION)
./mvn-packager.sh
docker build . -t labshare/polus-tiledtiff-converter-plugin:${version}
