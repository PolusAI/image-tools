#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/file-renaming-plugin:${version}
