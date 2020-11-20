#!/bin/bash

version=$(<VERSION)

# Python builds
docker build . -f ./docker/alpine/DockerfilePython -t labshare/polus-bfio-util:${version} -t labshare/polus-bfio-util:${version}-alpine -t labshare/polus-bfio-util:${version}-python -t labshare/polus-bfio-util:${version}-alpine-python
docker build . -f ./docker/slim-buster/DockerfilePython -t labshare/polus-bfio-util:${version}-slim-buster -t labshare/polus-bfio-util:${version}-slim-buster-python
docker build . -f ./docker/imagej/DockerfilePython -t labshare/polus-bfio-util:2.0.0a2-imagej
docker build . -f ./tensorflow/Dockerfile -t labshare/polus-bfio-util:${version}-tensorflow
