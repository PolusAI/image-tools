#!/bin/bash

version=$(<VERSION)
 docker build . -t labshare/polus-vector-label-plugin:${version}