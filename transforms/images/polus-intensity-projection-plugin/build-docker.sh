#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/intensity-projection-plugin:${version}