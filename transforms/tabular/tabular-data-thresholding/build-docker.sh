version=$(<VERSION)
docker build . -t polusai/tabular-data-thresholding:${version}