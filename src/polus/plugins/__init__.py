"""Initialize polus-plugins module."""

import logging

from polus.plugins._plugins.classes import (  # noqa # pylint: disable=unused-import
    get_plugin,
    list_plugins,
    load_plugin,
    refresh,
    submit_plugin,
)
from polus.plugins._plugins.update import (  # noqa # pylint: disable=unused-import
    update_nist_plugins,
    update_polus_plugins,
)

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

# for _p in __plugins:
#     # make each plugin available as polus.plugins.PluginName
#     globals()[_p] = get_plugin(_p)

# plugin_list = list_plugins()


def __getattr__(name):
    if name == "list":
        return __plugins
    if name in __plugins:
        return get_plugin(name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "refresh",
    "submit_plugin",
    "get_plugin",
    "load_plugin",
    "list_plugins",
    "update_polus_plugins",
    "update_nist_plugins",
]
