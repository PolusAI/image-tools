#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/nyxus-plugin:${version}
