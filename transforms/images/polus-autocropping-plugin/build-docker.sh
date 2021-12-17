#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-autocropping-plugin:"${version}"
