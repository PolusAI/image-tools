#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/precompute-slide-plugin:${version}

# Build for arm using buildx
# docker buildx build --load --platform linux/arm64/v8 --tag polusai/precompute-slide-plugin:${version} .

# Muli-platform deployment for the container (does not work currently)
# docker buildx build --platform linux/arm64,linux/arm64/v8 --tag polusai/precompute-slide-plugin:${version} 

# Muli-platform deployment for the container (does not work currently)
# docker buildx build --platform linux/arm64,linux/arm64 --tag polusai/precompute-slide-plugin:${version} --attest type=provenance,mode=min .
# docker buildx build --load -t polusai/precompute-slide-plugin:${version}  .