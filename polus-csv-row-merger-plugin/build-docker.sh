#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-csv-row-merger-plugin:${version}