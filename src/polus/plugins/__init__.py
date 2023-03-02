import logging

# TODO try to get rid of _Plugins
from polus.plugins._plugins.classes import _Plugins as plugins, submit_plugin

"""
Set up logging for the module
"""
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins")
plugins.refresh()  # calls the refresh method when library is imported

list = plugins.list

__all__ = ["plugins", "submit_plugin"]
