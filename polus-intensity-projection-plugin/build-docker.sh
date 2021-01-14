#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-intensity-projection-plugin-plugin:${version}