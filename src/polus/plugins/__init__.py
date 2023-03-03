"""Initialize polus-plugins module."""

import logging

from polus.plugins._plugins.classes import (  # noqa # pylint: disable=unused-import
    get_plugin, list_plugins, load_plugin, refresh)
from polus.plugins._plugins.update import (  # noqa # pylint: disable=unused-import
    update_nist_plugins, update_polus_plugins)

"""
Set up logging for the module
"""
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins")
refresh()  # calls the refresh method when library is imported
__plugins = list_plugins()

for _p in __plugins:
    # make each plugin available as polus.plugins.PluginName
    globals()[_p] = get_plugin(_p)

plugin_list = list_plugins()

_export_list = [
    "plugin_list",
    "refresh",
    "submit_plugin",
    "get_plugin",
    "load_plugin",
    "list_plugins",
    "update_polus_plugins",
    "update_nist_plugins",
] + __plugins

__all__ = _export_list
