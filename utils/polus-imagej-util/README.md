# Imagej Util

The Imagej util pipeline is used to create WIPP plugins from the [Imagej image processing operations (ops)](https://github.com/imagej/tutorials/tree/master/notebooks/1-Using-ImageJ). As of now this pipeline had the ability to generate approximately 92 different plugins with a total of 253 overloading methods. Since the Imagej ops were written in Java each op has a number of different overloading methods for different data type inputs. When an op is called the appropriate overloading method is used based upon the input data types. However, it should be noted that this pipeline is still under development and about 10% currently pass the automatically generated unit tests. In order for a op to be generated at least one of its overloading methods must be currently supported. Below are the criteria for op overloading method generation.


- All data types of the required inputs must map from a WIPP data type to an Imagej data type. Not all data type conversions are currently supported in this version. 
- The output data type must map from an Imagej data type to a WIPP data type.
- At least one of the required inputs must map to a `collection` data type.

In addition to the above criteria, at this time only the required inputs of the Imagej ops can be used when the op is called. The optional inputs are documented in the log files but not used when generating plugins. 

## How to use

1. Install cookiecutter: `pip install cookiecutter`
2. Clone `polus-plugins` and change to the polus-plugins directory
3. Modify `generate.py` to include author information
4. Run `generate.py` or modify `generate.py` to generate desired plugins


** NOTE: ** Do not modify `project_slug`. This is automatically generated from the name of the plugin. If the imagej plugin is called "Imagej Image Integral", then the plugin directory and docker container will have the name `polus-imagej-image-integral-plugin`.

## Explanation of File Structure and Settings

### General Structure

In general, the structure of automatically generated Imagej plugin should have the following files as a minimum:

```
plugin-root/
    - VERSION*
    - build-docker.sh
    - Dockerfile*
    - README.md*
    - plugin.json*
    - run-plugin.sh
    - src/
        - __init__.py
        - ij_converter.py
        - main.py
    - tests
        - __init__.py
        - unit_test.py
        - version_test.py
```

Files with a `*` at the end indicate files that are necessary. If none of the other files are modified, there are some built in tools to simplify containerization and deployment of existing code to WIPP.

### VERSION

This indicates the version of the plugin. It should follow the [semantic versioning](https://semver.org/) standard (for example `2.0.0`). The only thing that should be in this file is the version. The cookie cutter template defaults to the current version of this plugin generation pipeline, `0.2.0`.

This file is used to tag the docker container built with the `build-docker.sh` script and by Jenkins if the plugin is merged into `labshare/polus-plugins`.

### Dockerfile

This is a basic dockerfile. In general, this should contain all the necessary tools to run a basic Python plugin.

The Dockerfile uses Alpine Linux with Python 3.7 installed on it as the base image ([python:3.7-alpine](https://hub.docker.com/_/python)). If `use_bfio` is set to true, then `labshare/polus-bfio-util` is used, which uses Alpine Linux installed with Python 3 and OpenJDK 8 with the `bfio`, `javabridge`, `python-bioformats`, and `numpy` packages pre-installed.

For more information, check out the repositories for [javabridge](https://github.com/LeeKamentsky/python-javabridge), [python-bioformats](https://github.com/CellProfiler/python-bioformats), and [bfio](https://github.com/Nicholas-Schaub/polus-plugins/tree/master/utils/polus-bfio-util).

### README.md

A basic description of what the plugin does. This should define all the inputs and outputs. Cookiecutter should autogenerate the input and output table, but double check to make sure.

### plugin.json

This file defines the input and output variables for WIPP, and defines the UI components showed to the user. This should be automatically generated for basic variable types, but may need to be modified to work properly.

### build-docker.sh

This file builds a docker container using the name of the plugin and a tag using `VERSION`.

### src/main.py

This is the file called from the commandline from the docker container. Cookie cutter autogenerates all of the code required to build a Java Virtual Machine (JVM), call the op and return the output. If the name of this file is changed, then the `Dockerfile` will need to be modified with the name of the new file.

### src/requirements.txt

This file should contain a list of the packages (including versions) that are used by the plugin. It is important to make this as simple as possible. Since the base images are running on `alpine`, many commonly used packages need to be compiled. The `labshare/polus-bfio-tool` image comes with a compiled version of NumPy.

### tests/unit_test.py

This file is automatically generated during plugin generation, it is not intended to be run directly. Instead, it is run using a shell command, see Imagej Testing in the next section. 

## Imagej Testing

In the polus-plugins directory a new directory called `imagej-testing` is created when `genreate.py` runs. This file contains `shell_test.py` which has auto generated shell commands to run the `unit_test.py` file for all overloading methods of each op which was generated. Additionally, several logs will be placed here when running the unit tests. To begin testing run `shell_test.py` from the `polus-plugins` directory.

## Plugin Generation Pipeline Explanation

Each plugin is generated using the information from the Imagej ops help. From the help menu the inputs, outputs and function call can all be stored in a json file which is later used by `cookiecutter` to automatically create the plugin directory and required files to run the op. Below is a short high-level description of the entire pipeline.


1. `generate.py` utilizes `populate.py` classes to parse and store relevant information about each op and its overloading methods
2. `generate.py` calls `populate.py` class methods to create the `cookiecutter.json` files needed for plugin generation
3. `generate.py` utilizes `cookiecutter` to create the plugin directory and all required files
4. `main.py` is called from the command line when running or testing plugin and a JVM is started
5. `main.py` utilizes `ij_converter.py` to convert python data to Java data (WIPP types to Imagej types)
6. The op is called in `main.py`
7. `ij_converter` converts the output back from Imagej to WIPP
8. Finally `BioWriter` writes the output to the specified directory and the JVM is terminated.