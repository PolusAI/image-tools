"""Initialize polus-plugins module."""

import logging
from typing import Union

from polus.plugins._plugins.classes import (
    ComputePlugin,  # pylint: disable=unused-import
)
from polus.plugins._plugins.classes import Plugin  # pylint: disable=unused-import
from polus.plugins._plugins.classes import get_plugin  # pylint: disable=unused-import
from polus.plugins._plugins.classes import list_plugins  # pylint: disable=unused-import
from polus.plugins._plugins.classes import load_config  # pylint: disable=unused-import
from polus.plugins._plugins.classes import refresh  # pylint: disable=unused-import
from polus.plugins._plugins.classes import remove_all  # pylint: disable=unused-import
from polus.plugins._plugins.classes import (  # pylint: disable=unused-import
    remove_plugin,
)
from polus.plugins._plugins.classes import (  # pylint: disable=unused-import
    submit_plugin,
)
from polus.plugins._plugins.update import (  # pylint: disable=unused-import
    update_nist_plugins,
)
from polus.plugins._plugins.update import (  # pylint: disable=unused-import
    update_polus_plugins,
)

"""
Set up logging for the module
"""
logger = logging.getLogger("polus.plugins")


refresh()  # calls the refresh method when library is imported


def __getattr__(name: str) -> Union[Plugin, ComputePlugin, list]:
    if name == "list":
        return list_plugins()
    if name in list_plugins():
        return get_plugin(name)
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = [
    "refresh",
    "submit_plugin",
    "get_plugin",
    "load_config",
    "list_plugins",
    "update_polus_plugins",
    "update_nist_plugins",
    "remove_all",
    "remove_plugin",
]
