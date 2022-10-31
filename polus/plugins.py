import json
import typing
import logging
import re
from tqdm import tqdm
from pydantic import ValidationError
from .io import Version
from .plugin_classes import submit_plugin, _Plugins, load_plugin, Plugin, ComputePlugin
from .manifest_utils import _scrape_manifests, _error_log
from .gh import _init_github, add_plugin_from_gh
from .registry import WippPluginRegistry
from .utils import name_cleaner

"""
Set up logging for the module
"""
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins")
logger.setLevel(logging.INFO)
plugins = _Plugins()
get_plugin = plugins.get_plugin
load_config = plugins.load_config
plugins.WippPluginRegistry = WippPluginRegistry
plugins.refresh()  # calls the refresh method when library is imported


def update_polus_plugins(
    gh_auth: typing.Optional[str] = None, min_depth: int = 2, max_depth: int = 3
):

    logger.info("Updating polus plugins.")
    # Get all manifests
    valid, invalid = _scrape_manifests(
        "polusai/polus-plugins", _init_github(gh_auth), min_depth, max_depth, True
    )
    manifests = valid.copy()
    manifests.extend(invalid)
    logger.info("Submitting %s plugins." % len(manifests))

    for manifest in manifests:

        try:
            plugin = submit_plugin(manifest)

            """ Parsing checks specific to polus-plugins """
            error_list = []

            # Check that plugin version matches container version tag
            container_name, version = tuple(plugin.containerId.split(":"))
            version = Version(version=version)
            organization, container_name = tuple(container_name.split("/"))
            try:
                assert (
                    plugin.version == version
                ), f"containerId version ({version}) does not match plugin version ({plugin.version})"
            except AssertionError as err:
                error_list.append(err)

            # Check to see that the plugin is registered to Labshare
            try:
                assert organization in [
                    "polusai",
                    "labshare",
                ], "All polus plugin containers must be under the Labshare organization."
            except AssertionError as err:
                error_list.append(err)

            # Checks for container name, they are somewhat related to our
            # Jenkins build
            try:
                assert container_name.startswith(
                    "polus"
                ), "containerId name must begin with polus-"
            except AssertionError as err:
                error_list.append(err)

            try:
                assert container_name.endswith(
                    "plugin"
                ), "containerId name must end with -plugin"
            except AssertionError as err:
                error_list.append(err)

            if len(error_list) > 0:
                raise ValidationError(error_list, plugin.__class__)

        except ValidationError as val_err:
            try:
                _error_log(val_err, manifest, "update_polus_plugins")
            except BaseException as e:
                # logger.debug(f"There was an error {e} in {plugin.name}")
                logger.exception(f"In {plugin.name}: {e}")
        except BaseException as e:
            # logger.debug(f"There was an error {e} in {plugin.name}")
            logger.exception(f"In {plugin.name}: {e}")


def update_nist_plugins(gh_auth: typing.Optional[str] = None):

    # Parse README links
    gh = _init_github(gh_auth)
    repo = gh.get_repo("usnistgov/WIPP")
    contents = repo.get_contents("plugins")
    readme = [r for r in contents if r.name == "README.md"][0]
    pattern = re.compile(
        r"\[manifest\]\((https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))\)"
    )
    matches = pattern.findall(str(readme.decoded_content))
    logger.info("Updating NIST plugins.")
    for match in tqdm(matches, desc="NIST Manifests"):
        url_parts = match[0].split("/")[3:]
        plugin_repo = gh.get_repo("/".join(url_parts[:2]))
        manifest = json.loads(
            plugin_repo.get_contents("/".join(url_parts[4:])).decoded_content
        )

        try:
            submit_plugin(manifest)

        except ValidationError as val_err:
            _error_log(val_err, manifest, "update_nist_plugins")
