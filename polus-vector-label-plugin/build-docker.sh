#!/bin/bash

version=$(<VERSION)
sudo docker build . -t labshare/polus-cellpose--plugin:${version}