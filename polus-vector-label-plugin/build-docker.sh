#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-cellpose--plugin:${version}