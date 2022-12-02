#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-macro-plugin:${version}