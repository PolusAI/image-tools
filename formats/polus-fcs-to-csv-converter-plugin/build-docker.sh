#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/fcs-to-csv-converter-plugin:${version}