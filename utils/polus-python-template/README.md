# WIPP Plugin Cookie Cutter (for Python) (1.1.0-dev0)

This repository is a cookie cutter template that creates the basic scaffold structure of a
polus plugin and add it to the polus plugins directory structure.

## How to use
1. Clone `polus-plugins` and change to the polus-plugins directory
2. `cd /utils/polus-python-template/`
3. (optional) Install poetry if not available.
4. (optional) Create a dedicated environment with conda or venv.
5.  Install the dependencies: `poetry install`
6. Ignore changes to `cookiecutter.json` using: `git update-index --assume-unchanged cookiecutter.json`
7. Modify `cookiecutter.json` to include author and plugin information.`plugin_package` should always start with `polus.plugins`. 
** NOTE: ** Do not edit values in brackets ({}) as they are edited by cookiecutter directly.
Those are automatically generated from the previous entries. If your plugin is called 
"Awesome Function", then the plugin folder and docker container will have the name `awesome-function-plugin`.
8. Create your plugin skeleton: ` python -m cookiecutter . --no-input`


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


## Executing the plugin

The plugin should be run as a package.
To install the package :

`pip install .`

The skeleton code can be run this way :
From the plugin's top directory (with the default values):

`python -m polus.plugins1.package1.package2.awesome_function -i /tmp/inp -o /tmp/out`

This should print some logs with the provided inputs and outputs and return.

## Running tests
Plugin's developer should use `pytest`.
Some simple tests have been added to the template as examples.
Before submitting a PR to `polus-plugins`, other unit tests should be created and added to the `tests`
directory.

To run tests :

From the plugin's top directory, type `python -m pytest`.
Depending on how you have set up your environment, you may be able to run the pytest cli directly `pytest`. See pytest doc for how the project source directory is scanned to collect tests.
This should run a test successfully and return.


## Creating and running a container

` ./build-docker.sh &&  ./run-plugin.sh`

Build the docker image and run the container.

### DockerFile
A docker image is build from a base image with common dependencies pre-installed.
The image entrypoint will run the plugin's package entrypoint.

### build-docker.sh
Run this script to build the container.

### run-plugin.sh
Run the container locally.


## Customize the plugin

### Project code

A set of common dependencies are added to `pyproject.toml`.
Update according to your needs.

### Managing versioning

Making sure that the file version is consistent across files in a plugin can be
challenging, so the Python template now uses
[bump2version](https://github.com/c4urself/bump2version)
to help manage versioning. This automatically changes the `VERSION` and
`plugin.json` files to the next version, preventing you from having to remember
to change the version everywhere. The `bumpversion.cfg` can be modified to
change the version in other files as well.

To use this feature:
`bump2version --config-file bumpversion.cfg`

### Documentation

#### README.md

A basic description of what the plugin does. This should define all the inputs
and outputs.

#### CHANGELOG.md

Documents updates made to the plugin.


### WIPP manifest (plugin.json).

This file defines the input and output variables for WIPP, and defines the UI
components showed to the user.
