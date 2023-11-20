#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/notebook-plugin:${version}
