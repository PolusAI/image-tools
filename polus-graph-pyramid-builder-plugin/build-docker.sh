#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/labshare:${version}
