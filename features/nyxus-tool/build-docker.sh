#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/nyxus-tool:${version}
