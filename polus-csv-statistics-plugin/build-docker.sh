#!/bin/bash

version=$(<VERSION)
sudo docker build . -t labshare/polus-csv-statistics-plugin:${version}