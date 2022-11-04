#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/csv-statistics-plugin:${version}