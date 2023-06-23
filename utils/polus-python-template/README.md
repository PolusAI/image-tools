# WIPP Plugin Cookie Cutter (for Python) (1.1.0-dev0)

This repository is a cookie cutter template that gives the basic structure of a
polus plugin, and it's specially tailored to Python and Linux. However, even if
the code base for the plugin is not Python, it is still useful for generating
the basic skeleton of a plugin.

## How to use
1. Clone `polus-plugins` and change to the polus-plugins directory
2. `cd /utils/polus-python-template/`
3. Install the requirements: `poetry update`
4. Ignore changes to `cookiecutter.json` using: `git update-index --assume-unchanged ./utils/polus-python-template/cookiecutter.json`
5. Modify `cookiecutter.json` to include author and plugin information.`plugin_package` should always start with `polus.plugins`.
6. Create your plugin skeleton: ` python -m cookiecutter . --no-input`

** NOTE: ** Do not modify entries after `package_name`. Those are automatically generated from
the previous entries. If your plugin is called "Awesome Function", then
the plugin folder and docker container will have the name `awesome-function-plugin`.

## Plugin Standard
The generated plugin will be compatible with polus most up-to-date guidelines :
see [standard guidelines](https://labshare.atlassian.net/wiki/spaces/WIPP/pages/3275980801/Python+Plugin+Standards)

The code generated provides out-of-box support for :
    - customizing the plugin code.
    - implementing tests.
    - creating and running a container.
    - managing versioning.
    - updating documentation (README, CHANGELOG).
    - maintaining a WIPP manifest (plugin.json).

## Customizing the plugin code

The plugin should be run as a package.
The skeleton code can be run this way :
From the plugin's top directory :
`cd src`
`python -m polus.package1.package2.awesome_function -i /tmp/inp -o /tmp/out`

This should print some logs with the provided inputs and outputs and return.

## Implementing tests

Plugin's developer should preferably uses `pytest`.
A simple test has been added to the template as an example.
Before submitting a PR to `polus-plugins`, other unit tests should be created and added to the `tests`
directory.

To run tests :
From the plugin's top directory, type `python -m pytest`.
This should run a test successfully and return.


## Managing versioning

Making sure that the file version is consistent across files in a plugin can be
challenging, so the Python template now uses
[bump2version](https://github.com/c4urself/bump2version)
to help manage versioning. This automatically changes the `VERSION` and
`plugin.json` files to the next version, preventing you from having to remember
to change the version everywhere. The `bumpversion.cfg` can be modified to
change the version in other files as well.

To use this feature:
1. `pip install bump2version`
2. `bump2version --config-file bumpversion.cfg`
3. If an error is thrown about uncommited work, use the `--allow-dirty` option

## Creating and running a container

` ./build-docker.sh &&  ./run-plugin.sh`

Should build the docker image and run it successfully.

### DockerFile
A docker image is build from a base image with common dependencies pre-installed.
The image entrypoint will run the plugin's package entrypoint.

### build-docker.sh
Run this script to build the container.

### run-plugin.sh
Run the container locally.

## Updating Documentation
### README.md

A basic description of what the plugin does. This should define all the inputs
and outputs.

### CHANGELOG.md

Documents updates made to the plugin.

## Maintaining a WIPP manifest (plugin.json).
### plugin.json

This file defines the input and output variables for WIPP, and defines the UI
components showed to the user.
