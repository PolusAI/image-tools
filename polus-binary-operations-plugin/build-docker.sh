#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-binary-operations-plugin:${version}
