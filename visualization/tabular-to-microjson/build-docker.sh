#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tabular-to-microjson-plugin:${version}
