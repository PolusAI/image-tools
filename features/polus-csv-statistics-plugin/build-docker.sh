#!/bin/bash

version=$(<VERSION)
 docker build . -t labshare/polus-csv-statistics-plugin:${version}