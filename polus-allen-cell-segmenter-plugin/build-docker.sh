#!/bin/bash

version=$(<VERSION)
docker build . -t gauhar2020/sample_plugin:${version}