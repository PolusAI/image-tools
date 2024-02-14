#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/image-calculator-plugin:"${version}"
