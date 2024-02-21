"""Initialize update module."""

from polus.plugins._plugins.update._update import update_nist_plugins
from polus.plugins._plugins.update._update import update_polus_plugins

__all__ = ["update_polus_plugins", "update_nist_plugins"]
