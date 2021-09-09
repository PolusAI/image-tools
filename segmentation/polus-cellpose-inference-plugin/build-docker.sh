#!/bin/bash

version=$(<VERSION)
docker build . -f ./Dockerfile -t labshare/polus-cellpose-inference-plugin:"${version}"