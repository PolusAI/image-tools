#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/lumos-bleedthrough-correction-tool:"${version}"
