#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-subset-feature-data-plugin:${version}