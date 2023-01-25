#!/bin/bash

version=$(grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
docker push polusai/roi-relabel-plugin:"${version}"
