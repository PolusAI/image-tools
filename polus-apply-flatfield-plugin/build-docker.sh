#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-apply-flatfield-plugin:${version}