#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-notebook-plugin:${version}
