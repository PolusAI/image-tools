#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-transform-flatiterableview-plugin:${version}