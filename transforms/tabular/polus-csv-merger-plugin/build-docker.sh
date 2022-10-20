#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-csv-merger-plugin:${version}