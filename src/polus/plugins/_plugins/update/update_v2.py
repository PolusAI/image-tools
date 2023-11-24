# pylint: disable=W1203, W1201
import json
import logging
import re
import typing

from polus.plugins._plugins.classes import refresh
from polus.plugins._plugins.classes import submit_plugin
from polus.plugins._plugins.gh import _init_github
from polus.plugins._plugins.io import Version
from polus.plugins._plugins.manifests import _error_log
from polus.plugins._plugins.manifests import _scrape_manifests
from pydantic import ValidationError
from tqdm import tqdm

logger = logging.getLogger("polus.plugins")


def update_polus_plugins(
    gh_auth: typing.Optional[str] = None,
    min_depth: int = 2,
    max_depth: int = 3,
) -> None:
    """Scrape PolusAI GitHub repo and create local versions of Plugins."""
    logger.info("Updating polus plugins.")
    # Get all manifests
    valid, invalid = _scrape_manifests(
        "polusai/polus-plugins",
        _init_github(gh_auth),
        min_depth,
        max_depth,
        True,
    )
    manifests = valid.copy()
    manifests.extend(invalid)
    logger.info(f"Submitting {len(manifests)} plugins.")

    for manifest in manifests:
        try:
            plugin = submit_plugin(manifest)

            # Parsing checks specific to polus-plugins
            error_list = []

            # Check that plugin version matches container version tag
            container_name, version = tuple(plugin.containerId.split(":"))
            version = Version(version)
            organization, container_name = tuple(container_name.split("/"))
            if plugin.version != version:
                msg = (
                    f"containerId version ({version}) does not "
                    f"match plugin version ({plugin.version})"
                )
                logger.error(msg)
                error_list.append(ValueError(msg))

            # Check to see that the plugin is registered to Labshare
            if organization not in ["polusai", "labshare"]:
                msg = (
                    "all polus plugin containers must be"
                    " under the Labshare organization."
                )
                logger.error(msg)
                error_list.append(ValueError(msg))

            # Checks for container name, they are somewhat related to our
            # Jenkins build
            if not container_name.startswith("polus"):
                msg = "containerId name must begin with polus-"
                logger.error(msg)
                error_list.append(ValueError(msg))

            if not container_name.endswith("plugin"):
                msg = "containerId name must end with -plugin"
                logger.error(msg)
                error_list.append(ValueError(msg))

            if len(error_list) > 0:
                raise ValidationError(error_list, plugin.__class__)

        except ValidationError as val_err:
            try:
                _error_log(val_err, manifest, "update_polus_plugins")
            except BaseException as e:  # pylint: disable=W0718
                logger.exception(f"In {plugin.name}: {e}")
        except BaseException as e:  # pylint: disable=W0718
            logger.exception(f"In {plugin.name}: {e}")
    refresh()


def update_nist_plugins(gh_auth: typing.Optional[str] = None) -> None:
    """Scrape NIST GitHub repo and create local versions of Plugins."""
    # Parse README links
    gh = _init_github(gh_auth)
    repo = gh.get_repo("usnistgov/WIPP")
    contents = repo.get_contents("plugins")
    readme = [r for r in contents if r.name == "README.md"][0]
    pattern = re.compile(
        r"\[manifest\]\((https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))\)",
    )
    matches = pattern.findall(str(readme.decoded_content))
    logger.info("Updating NIST plugins.")
    for match in tqdm(matches, desc="NIST Manifests"):
        url_parts = match[0].split("/")[3:]
        plugin_repo = gh.get_repo("/".join(url_parts[:2]))
        manifest = json.loads(
            plugin_repo.get_contents("/".join(url_parts[4:])).decoded_content,
        )

        try:
            submit_plugin(manifest)

        except ValidationError as val_err:
            _error_log(val_err, manifest, "update_nist_plugins")
    refresh()
