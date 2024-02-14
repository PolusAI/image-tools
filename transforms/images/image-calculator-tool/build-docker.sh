#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/image-calculator-tool:"${version}"
