#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-color-pyramid-builder-plugin:${version}