
version=$(<VERSION)
docker build . -t labshare/polus-generic-to-image-collection-plugin:${version}