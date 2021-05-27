#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-hdbscan-clustering-plugin:${version}