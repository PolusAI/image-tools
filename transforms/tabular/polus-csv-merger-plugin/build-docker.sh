#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/csv-merger-plugin:${version}