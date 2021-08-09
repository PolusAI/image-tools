#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-label-to-vector-plugin:${version}