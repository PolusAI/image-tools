# WIPP Plugin Cookie Cutter (for Python)

This repository is a cookie cutter template that gives the basic structure of a WIPP plugin, and it's specially tailored to Python and Linux. However, even if the code base for the plugin is not Python, it is still useful for generating a basic skeleton of a plugin.

## How to use

1. Install cookiecutter: `pip install cookiecutter`
2. Clone `polus-plugins` and change to the polus-plugins directory
3. Ignore changes to `cookiecutter.json` using: `git update-index --assume-unchanged ./utils/polus-python-template/cookiecutter.json`
4. Modify `cookiecutter.json` to include author and plugin information.
5. Create your plugin skeleton: `cookiecutter ./utils/polus-python-template/ --no-input`

** NOTE: ** Do not modify `project_slug`. This is automatically generated from the name of the plugin. If your plugin is called "Awesome Segmentation", then the plugin folder and docker container will have the name `polus-awesome-segmentation-plugin`.