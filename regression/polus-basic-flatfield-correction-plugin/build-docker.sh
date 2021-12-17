#!/bin/bash
version=$(<VERSION)
docker build . -t labshare/polus-basic-flatfield-correction-plugin:${version}