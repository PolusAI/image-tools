#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/graph-pyramid-builder:${version}