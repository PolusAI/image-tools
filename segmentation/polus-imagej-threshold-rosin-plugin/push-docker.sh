#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-rosin-plugin:${version}