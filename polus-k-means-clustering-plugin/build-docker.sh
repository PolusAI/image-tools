#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-feature-extraction-plugin:${version}