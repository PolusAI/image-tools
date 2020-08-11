#!/bin/bash

version=$(<VERSION)
docker build . -f ./alpine/Dockerfile -t labshare/polus-bfio-util:${version}-alpine -t labshare/polus-bfio-util:${version}
docker build . -f ./slim-buster/Dockerfile -t labshare/polus-bfio-util:${version}-slim-buster
docker build . -f ./tensorflow/Dockerfile -t labshare/polus-bfio-util:${version}-tensorflow
