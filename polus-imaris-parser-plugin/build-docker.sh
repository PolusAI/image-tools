#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imaris-parser-plugin:${version}