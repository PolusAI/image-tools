version=$(<VERSION)
docker build . -t polusai/tabular-thresholding-tool:${version}