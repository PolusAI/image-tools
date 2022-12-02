#!/bin/bash
version=$(<VERSION)
./mvn-packager.sh
docker build . -t polusai/tiledtiff-converter-plugin:${version}
