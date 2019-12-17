#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-csv-merger-plugin:${version}