#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-feature-data-subset-plugin:${version}