#!/bin/bash

version=$(<VERSION)
docker push polusai/montage-plugin:${version}