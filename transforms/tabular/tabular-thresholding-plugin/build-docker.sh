version=$(<VERSION)
docker build . -t polusai/tabular-thresholding-plugin:${version}