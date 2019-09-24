#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-bfio-util:${version}