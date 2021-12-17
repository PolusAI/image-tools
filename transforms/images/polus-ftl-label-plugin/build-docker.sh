#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-ftl-label-plugin:"${version}"