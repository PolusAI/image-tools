#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-image-registration-plugin:${version}