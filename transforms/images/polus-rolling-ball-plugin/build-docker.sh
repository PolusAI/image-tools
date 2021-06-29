#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-rolling-ball-plugin:"${version}"
