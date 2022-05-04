#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-remove-border-objects-plugin:${version}