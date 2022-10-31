import logging
from urllib.parse import urljoin
import github
import os
from ..plugin_classes import submit_plugin

logger = logging.getLogger("polus.plugins")

"""
Initialize the Github interface
"""


def _init_github(auth=None):

    if auth is None:

        # Try to get an auth key from an environment variable
        auth = os.environ.get("GITHUB_AUTH", None)

        if auth is None:
            gh = github.Github()
            logger.warning("Initialized Github connection with no user token.")
            return gh
        else:
            logger.debug("Found auth token in GITHUB_AUTH environment variable.")

    else:
        logger.debug("Github auth token supplied as input.")

    gh = github.Github(login_or_token=auth)
    logger.debug(
        f"Initialized Github connection with token for user: {gh.get_user().login}"
    )

    return gh


def add_plugin_from_gh(
    user: str,
    branch: str,
    plugin: str,
    repo: str = "polus-plugins",
    manifest_name: str = "plugin.json",
):
    """Add plugin from GitHub.

    This function adds a plugin hosted on GitHub and returns a Plugin object.

    Args:
        user: GitHub username
        branch: GitHub branch
        plugin: Plugin's name
        repo: Name of GitHub repository, default is `polus-plugins`
        manifest_name: Name of manifest file, default is `plugin.json`

    Returns:
        A Plugin object populated with information from the plugin manifest.
    """
    l = [user, repo, branch, plugin, manifest_name]
    u = "/".join(l)
    url = urljoin("https://raw.githubusercontent.com", u)
    logger.info("Adding %s" % url)
    return submit_plugin(url, refresh=True)
