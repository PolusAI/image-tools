#!/bin/bash

version=$(<VERSION)
docker push polusai/render-overlay-plugin:${version}