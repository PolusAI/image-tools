"""
Validate of template variables before templating the project
"""
import logging
from os import environ

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "DEBUG"))
logger = logging.getLogger("polus-python-template-pre")
logger.setLevel(POLUS_LOG)

# NOTE Those validation could be performed on a plugin.json
# using polus plugins pydantic models.

author = "{{ cookiecutter.author }}"
# TODO check valid

author_email = "{{ cookiecutter.author_email }}"
## TODO check valid

plugin_package = "{{ cookiecutter.plugin_package }}"
if not plugin_package.startswith("polus.plugins."):
    raise ValueError(
        f"plugin package must be a child of polus.plugins."
        + f"plugin_package must start with 'polus.plugins'. Got : {plugin_package}"
    )
if plugin_package.endswith("_plugin"):
    raise ValueError(
        f"plugin_package must not ends with _plugin. Got : {plugin_package}"
    )

# TODO check we have a valid python package name

plugin_version = "{{ cookiecutter.plugin_version }}"
# TODO check version is valid

project_name = "{{ cookiecutter.project_name }}"
assert not ("_" in project_name) and not ("." in project_name)

plugin_slug = "{{ cookiecutter.plugin_slug }}"
assert plugin_slug.startswith("polus-") and plugin_slug.endswith("-plugin")

container_name = "{{ cookiecutter.container_name }}"
assert container_name.endswith("-plugin")

container_id = "{{ cookiecutter.container_id }}"
assert container_id.startswith("polusai/")

container_version = "{{ cookiecutter.container_version }}"
assert container_version == plugin_version

logger.debug(f"plugin_package: {plugin_package}" )
