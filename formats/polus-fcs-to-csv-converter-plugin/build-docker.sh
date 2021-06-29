#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-fcs-to-csv-converter-plugin:${version}