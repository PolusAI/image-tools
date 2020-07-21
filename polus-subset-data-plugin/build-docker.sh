#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-subset-data-plugin:${version}