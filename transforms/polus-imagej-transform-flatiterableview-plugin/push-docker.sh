#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-transform-flatiterableview-plugin:${version}