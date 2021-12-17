# WIPP Plugin Cookie Cutter (for Python)

This repository is a cookie cutter template that gives the basic structure of a
WIPP plugin, and it's specially tailored to Python and Linux. However, even if
the code base for the plugin is not Python, it is still useful for generating
a basic skeleton of a plugin.

## How to use

1. Clone `polus-plugins` and change to the polus-plugins directory
2. Install the requirements: `pip install -r ./utils/polus-python-template/requirements.txt`
3. Ignore changes to `cookiecutter.json` using: `git update-index --assume-unchanged ./utils/polus-python-template/cookiecutter.json`
4. Modify `cookiecutter.json` to include author and plugin information.
5. Create your plugin skeleton: `cookiecutter ./utils/polus-python-template/ --no-input`

** NOTE: ** Do not modify `project_slug`. This is automatically generated from
the name of the plugin. If your plugin is called "Awesome Segmentation", then
the plugin folder and docker container will have the name
`polus-awesome-segmentation-plugin`.

## What's New?

## bump2version

Making sure that the file version is consistent across files in a plugin can be
challenging, so the Python template now uses
[bump2version](https://github.com/c4urself/bump2version)
to help manage versioning. This automatically changes the `VERSION` and
`plugin.json` files to the next version, preventing you from having to remember
to change the version in all 3 places. The `bumpversion.cfg` can be modified to
change the version in other files as well.

To use this feature:
1. `pip install bump2version`
2. `bump2version --config-file bumpversion.cfg`
3. If an error is thrown about uncommited work, use the `--alow-dirty` option

## unittest

A simple `unittest` has been added to the template that verifies the `VERSION`
matches the `plugin.json` version information. Before submitting a PR to
`polus-plugins`, other unit tests should be created and added to the `tests`
directory. The test classes that are created should be added to the tuple
`test_cases` in `tests/__init__.py`.

## run-plugin.sh

A new shell script (`run-plugin.sh`) has been added to help setup testing of
docker containers locally. It comes configured with the proper `docker run`
command already set up so that input variables just need to be defined.

## Explanation of File Structure and Settings

### General Structure

In general, the structure of a plugin should have the following files as a minimum:

```
plugin-root/
    - VERSION*
    - build-docker.sh
    - run-plugin.sh
    - bumpversion.cfg
    - Dockerfile*
    - README.md*
    - plugin.json*
    - src/
        - main.py
        - requirements.txt
    - tests/
        - __init__.py
        - version_test.py
```

Files with a `*` at the end indicate files that are necessary. If none of the
other files are modified, there are some built in tools to simplify
containerization and deployment of existing code to WIPP.

### VERSION

This indicates the version of the plugin. It should follow the
[semantic versioning](https://semver.org/) standard (for example `2.0.0`). The
only thing that should be in this file is the version. The cookie cutter
template defaults to `0.1.0`.

This file is used to tag the docker container built with the `build-docker.sh`
script and by Jenkins if the plugin is merged into `labshare/polus-plugins`.

### Dockerfile

This is a basic dockerfile. In general, this should contain all the necessary
tools to run a basic Python plugin.

The Dockerfile uses Debian Linux (slim buster) with Python 3.9 installed on it
as the base image ([python:3.9-slim](https://hub.docker.com/_/python)).

If `use_bfio` is set to true, then `labshare/polus-bfio-util` is used as the
base image.

If `use_java` is set to true, then the container uses a version of
`labshare/polus-bfio-util` that comes with Java, javabridge, and bioformats.

If `use_filepattern` is set to true, then `requirements.txt` will be generated
to include `filepattern` and `main.py` will have some boilerplate code for
using the `filepattern` package.

For more information, check out the repositories for
[javabridge](https://github.com/LeeKamentsky/python-javabridge),
[python-bioformats](https://github.com/CellProfiler/python-bioformats),
and [bfio](https://github.com/Nicholas-Schaub/polus-plugins/tree/master/utils/polus-bfio-util).

### README.md

A basic description of what the plugin does. This should define all the inputs
and outputs. Cookiecutter should autogenerate the input and output table, but
double check to make sure.

### plugin.json

This file defines the input and output variables for WIPP, and defines the UI
components showed to the user. This should be automatically generated for basic
variable types, but may need to be modified to work properly.

### build-docker.sh

This file builds a docker container using the name of the plugin and a tag using
`VERSION`.

### src/main.py

This is the file called from the commandline from the docker container. Cookie
cutter autogenerates some basic code based on the inputs specified in the
cookiecutter json, for example there is code to parse commandline arguments. If
the name of this file is changed, then the `Dockerfile` will need to be modified
with the name of the new file.

### src/requirements.txt

This file should contain a list of the packages (including versions) that are
used by the plugin. It is important to make this as simple as possible.
