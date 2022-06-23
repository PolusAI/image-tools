#!/bin/bash

version=$(<VERSION)

docker push polusai/bleed-through-estimation-plugin:"${version}"
