#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/mesmer-inference-tool:${version}
