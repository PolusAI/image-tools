#!/bin/bash

version=$(<VERSION)
sudo docker build . -t labshare/polus-vector-label-plugin:${version}