# This script is designed to help package a new version of a plugin

# Type of new prelease, must be one of patch, minor, major
BUMP_VERSION=patch

# Bump the version
bump2version --config-file bumpversion.cfg ${BUMP_VERSION} --allow-dirty

# Get the new version
version=$(<VERSION)

# Build the container
./build-docker.sh

# Push to dockerhub
docker push labshare/{{ cookiecutter.project_slug }}:${version}
