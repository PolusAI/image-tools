"""Plugin classes and functions."""
import pydantic

PYDANTIC_VERSION = pydantic.__version__

if PYDANTIC_VERSION.split(".")[0] == "1":
    from polus.plugins._plugins.classes.plugin_classes_v1 import PLUGINS
    from polus.plugins._plugins.classes.plugin_classes_v1 import ComputePlugin
    from polus.plugins._plugins.classes.plugin_classes_v1 import Plugin
    from polus.plugins._plugins.classes.plugin_classes_v1 import _load_plugin
    from polus.plugins._plugins.classes.plugin_classes_v1 import get_plugin
    from polus.plugins._plugins.classes.plugin_classes_v1 import list_plugins
    from polus.plugins._plugins.classes.plugin_classes_v1 import load_config
    from polus.plugins._plugins.classes.plugin_classes_v1 import refresh
    from polus.plugins._plugins.classes.plugin_classes_v1 import remove_all
    from polus.plugins._plugins.classes.plugin_classes_v1 import remove_plugin
    from polus.plugins._plugins.classes.plugin_classes_v1 import submit_plugin
elif PYDANTIC_VERSION.split(".")[0] == "2":
    from polus.plugins._plugins.classes.plugin_classes_v2 import PLUGINS
    from polus.plugins._plugins.classes.plugin_classes_v2 import ComputePlugin
    from polus.plugins._plugins.classes.plugin_classes_v2 import Plugin
    from polus.plugins._plugins.classes.plugin_classes_v2 import _load_plugin
    from polus.plugins._plugins.classes.plugin_classes_v2 import get_plugin
    from polus.plugins._plugins.classes.plugin_classes_v2 import list_plugins
    from polus.plugins._plugins.classes.plugin_classes_v2 import load_config
    from polus.plugins._plugins.classes.plugin_classes_v2 import refresh
    from polus.plugins._plugins.classes.plugin_classes_v2 import remove_all
    from polus.plugins._plugins.classes.plugin_classes_v2 import remove_plugin
    from polus.plugins._plugins.classes.plugin_classes_v2 import submit_plugin

__all__ = [
    "Plugin",
    "ComputePlugin",
    "submit_plugin",
    "get_plugin",
    "refresh",
    "list_plugins",
    "remove_plugin",
    "remove_all",
    "load_config",
    "_load_plugin",
    "PLUGINS",
]
