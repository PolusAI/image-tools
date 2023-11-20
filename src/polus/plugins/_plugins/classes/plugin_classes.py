"""Classes for Plugin objects containing methods to configure, run, and save."""
# pylint: disable=W1203, enable=W1201
import json
import logging
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

from polus.plugins._plugins.classes.plugin_methods import _PluginMethods
from polus.plugins._plugins.io import DuplicateVersionFound
from polus.plugins._plugins.io import Version
from polus.plugins._plugins.io import _in_old_to_new
from polus.plugins._plugins.io import _ui_old_to_new
from polus.plugins._plugins.manifests.manifest_utils import InvalidManifest
from polus.plugins._plugins.manifests.manifest_utils import _load_manifest
from polus.plugins._plugins.manifests.manifest_utils import validate_manifest
from polus.plugins._plugins.models import ComputeSchema
from polus.plugins._plugins.models import PluginUIInput
from polus.plugins._plugins.models import PluginUIOutput
from polus.plugins._plugins.models import WIPPPluginManifest
from polus.plugins._plugins.utils import cast_version
from polus.plugins._plugins.utils import name_cleaner
from pydantic import Extra

logger = logging.getLogger("polus.plugins")
PLUGINS: dict[str, dict] = {}
# PLUGINS = {"BasicFlatfieldCorrectionPlugin":
#               {Version('0.1.4'): Path(<...>), Version('0.1.5'): Path(<...>)}.
#            "VectorToLabel": {Version(...)}}

"""
Paths and Fields
"""
# Location to store any discovered plugin manifests
_PLUGIN_DIR = Path(__file__).parent.parent.joinpath("manifests")


def refresh() -> None:
    """Refresh the plugin list."""
    organizations = [
        x for x in _PLUGIN_DIR.iterdir() if x.name != "__pycache__" and x.is_dir()
    ]  # ignore __pycache__

    PLUGINS.clear()

    for org in organizations:
        for file in org.iterdir():
            if file.suffix == ".py":
                continue

            try:
                plugin = validate_manifest(file)
            except InvalidManifest:
                logger.warning(f"Validation error in {file!s}")
            except BaseException as exc:  # pylint: disable=W0718 # noqa: BLE001
                logger.warning(f"Unexpected error {exc} with {file!s}")

            else:
                key = name_cleaner(plugin.name)
                # Add version and path to VERSIONS
                if key not in PLUGINS:
                    PLUGINS[key] = {}
                if (
                    plugin.version in PLUGINS[key]
                    and file != PLUGINS[key][plugin.version]
                ):
                    msg = (
                        "Found duplicate version of plugin"
                        f"{plugin.name} in {_PLUGIN_DIR}"
                    )
                    raise DuplicateVersionFound(
                        msg,
                    )
                PLUGINS[key][plugin.version] = file


def list_plugins() -> list:
    """List all local plugins."""
    output = list(PLUGINS.keys())
    output.sort()
    return output


class Plugin(WIPPPluginManifest, _PluginMethods):
    """WIPP Plugin Class.

    Contains methods to configure, run, and save plugins.

    Attributes:
        versions: A list of local available versions for this plugin.

    Methods:
        save_manifest(path): save plugin manifest to specified path
    """

    id: uuid.UUID  # noqa: A003

    class Config:
        """Config class for Pydantic Model."""

        extra = Extra.allow
        allow_mutation = False

    def __init__(self, _uuid: bool = True, **data: dict) -> None:
        """Init a plugin object from manifest."""
        if _uuid:
            data["id"] = uuid.uuid4()  # type: ignore
        else:
            data["id"] = uuid.UUID(str(data["id"]))  # type: ignore

        data["version"] = cast_version(data["version"])
        super().__init__(**data)

        self.Config.allow_mutation = True
        self._io_keys = {i.name: i for i in self.inputs}
        self._io_keys.update({o.name: o for o in self.outputs})

        if not self.author:
            warn_msg = (
                f"The plugin ({self.name}) is missing the author field. "
                "This field is not required but should be filled in."
            )
            logger.warning(warn_msg)

    @property
    def versions(self) -> list:  # cannot be in PluginMethods because PLUGINS lives here
        """Return list of local versions of a Plugin."""
        return list(PLUGINS[name_cleaner(self.name)])

    def to_compute(
        self,
        hardware_requirements: Optional[dict] = None,
    ) -> type[ComputeSchema]:
        """Convert WIPP Plugin object to Compute Plugin object."""
        data = deepcopy(self.manifest)
        return ComputePlugin(
            hardware_requirements=hardware_requirements,
            _from_old=True,
            **data,
        )

    def save_manifest(
        self,
        path: Union[str, Path],
        hardware_requirements: Optional[dict] = None,
        compute: bool = False,
    ) -> None:
        """Save plugin manifest to specified path."""
        if compute:
            with Path(path).open("w", encoding="utf-8") as file:
                self.to_compute(
                    hardware_requirements=hardware_requirements,
                ).save_manifest(path)
        else:
            with Path(path).open("w", encoding="utf-8") as file:
                dict_ = self.manifest
                json.dump(
                    dict_,
                    file,
                    indent=4,
                )

        logger.debug(f"Saved manifest to {path}")

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set I/O parameters as attributes."""
        _PluginMethods.__setattr__(self, name, value)

    @property
    def _config_file(self) -> dict:
        config_ = self._config
        config_["class"] = "WIPP"
        return config_

    def save_config(self, path: Union[str, Path]) -> None:
        """Save manifest with configured I/O parameters to specified path."""
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(self._config_file, file, indent=4, default=str)
        logger.debug(f"Saved config to {path}")

    def __repr__(self) -> str:
        """Print plugin name and version."""
        return _PluginMethods.__repr__(self)


class ComputePlugin(ComputeSchema, _PluginMethods):
    """Compute Plugin Class.

    Contains methods to configure, run, and save plugins.

    Attributes:
        versions: A list of local available versions for this plugin.

    Methods:
        save_manifest(path): save plugin manifest to specified path
    """

    class Config:
        """Config class for Pydantic Model."""

        extra = Extra.allow
        allow_mutation = False

    def __init__(
        self,
        hardware_requirements: Optional[dict] = None,
        _from_old: bool = False,
        _uuid: bool = True,
        **data: dict,
    ) -> None:
        """Init a plugin object from manifest."""
        if _uuid:
            data["id"] = uuid.uuid4()  # type: ignore
        else:
            data["id"] = uuid.UUID(str(data["id"]))  # type: ignore

        if _from_old:

            def _convert_input(dict_: dict) -> dict:
                dict_["type"] = _in_old_to_new(dict_["type"])
                return dict_

            def _convert_output(dict_: dict) -> dict:
                dict_["type"] = "path"
                return dict_

            def _ui_in(dict_: dict) -> PluginUIInput:  # assuming old all ui input
                # assuming format inputs. ___
                inp = dict_["key"].split(".")[-1]  # e.g inpDir
                try:
                    type_ = [x["type"] for x in data["inputs"] if x["name"] == inp][
                        0
                    ]  # get type from i/o
                except IndexError:
                    type_ = "string"  # default to string
                except BaseException as exc:
                    raise exc

                dict_["type"] = _ui_old_to_new(type_)
                return PluginUIInput(**dict_)

            def _ui_out(dict_: dict) -> PluginUIOutput:
                new_dict_ = deepcopy(dict_)
                new_dict_["name"] = "outputs." + new_dict_["name"]
                new_dict_["type"] = _ui_old_to_new(new_dict_["type"])
                return PluginUIOutput(**new_dict_)

            data["inputs"] = [_convert_input(x) for x in data["inputs"]]  # type: ignore
            data["outputs"] = [
                _convert_output(x) for x in data["outputs"]
            ]  # type: ignore
            data["pluginHardwareRequirements"] = {}
            data["ui"] = [_ui_in(x) for x in data["ui"]]  # type: ignore
            data["ui"].extend(  # type: ignore[attr-defined]
                [_ui_out(x) for x in data["outputs"]],
            )

        if hardware_requirements:
            for k, v in hardware_requirements.items():
                data["pluginHardwareRequirements"][k] = v

        data["version"] = cast_version(data["version"])
        super().__init__(**data)
        self.Config.allow_mutation = True
        self._io_keys = {i.name: i for i in self.inputs}
        self._io_keys.update({o.name: o for o in self.outputs})  # type: ignore

        if not self.author:
            warn_msg = (
                f"The plugin ({self.name}) is missing the author field. "
                "This field is not required but should be filled in."
            )
            logger.warning(warn_msg)

    @property
    def versions(self) -> list:  # cannot be in PluginMethods because PLUGINS lives here
        """Return list of local versions of a Plugin."""
        return list(PLUGINS[name_cleaner(self.name)])

    @property
    def _config_file(self) -> dict:
        config_ = self._config
        config_["class"] = "Compute"
        return config_

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set I/O parameters as attributes."""
        _PluginMethods.__setattr__(self, name, value)

    def save_config(self, path: Union[str, Path]) -> None:
        """Save configured manifest with I/O parameters to specified path."""
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(self._config_file, file, indent=4)
        logger.debug(f"Saved config to {path}")

    def save_manifest(self, path: Union[str, Path]) -> None:
        """Save plugin manifest to specified path."""
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(self.manifest, file, indent=4)
        logger.debug(f"Saved manifest to {path}")

    def __repr__(self) -> str:
        """Print plugin name and version."""
        return _PluginMethods.__repr__(self)


def _load_plugin(
    manifest: Union[str, dict, Path],
) -> Union[Plugin, ComputePlugin]:
    """Parse a manifest and return one of Plugin or ComputePlugin."""
    manifest = _load_manifest(manifest)
    if "pluginHardwareRequirements" in manifest:  # type: ignore[operator]
        # Parse the manifest
        plugin = ComputePlugin(**manifest)  # type: ignore[arg-type]
    else:
        # Parse the manifest
        plugin = Plugin(**manifest)  # type: ignore[arg-type]
    return plugin


def submit_plugin(
    manifest: Union[str, dict, Path],
) -> Union[Plugin, ComputePlugin]:
    """Parse a plugin and create a local copy of it.

    This function accepts a plugin manifest as a string, a dictionary (parsed
    json), or a pathlib.Path object pointed at a plugin manifest.

    Args:
        manifest:
            A plugin manifest. It can be a url, a dictionary,
            a path to a JSON file or a string that can be parsed as a dictionary

    Returns:
        A Plugin object populated with information from the plugin manifest.
    """
    plugin = validate_manifest(manifest)
    plugin_name = name_cleaner(plugin.name)

    # Get Major/Minor/Patch versions
    out_name = (
        plugin_name
        + f"_M{plugin.version.major}m{plugin.version.minor}p{plugin.version.patch}.json"
    )

    # Save the manifest if it doesn't already exist in the database
    organization = plugin.containerId.split("/")[0]
    org_path = _PLUGIN_DIR.joinpath(organization.lower())
    org_path.mkdir(exist_ok=True, parents=True)
    if not org_path.joinpath(out_name).exists():
        with org_path.joinpath(out_name).open("w", encoding="utf-8") as file:
            manifest_ = plugin.dict()
            manifest_["version"] = manifest_["version"]["version"]
            json.dump(manifest_, file, indent=4)

    # Refresh plugins list
    refresh()
    return plugin


def get_plugin(
    name: str,
    version: Optional[str] = None,
) -> Union[Plugin, ComputePlugin]:
    """Get a plugin with option to specify version.

    Return a plugin object with the option to specify a version.
    The specified version's manifest must exist in manifests folder.

    Args:
        name: Name of the plugin.
        version: Optional version of the plugin, must follow semver.

    Returns:
        Plugin object
    """
    if version is None:
        return _load_plugin(PLUGINS[name][max(PLUGINS[name])])
    return _load_plugin(PLUGINS[name][Version(**{"version": version})])


def load_config(config: Union[dict, Path]) -> Union[Plugin, ComputePlugin]:
    """Load configured plugin from config file/dict."""
    if isinstance(config, Path):
        with config.open("r", encoding="utf-8") as file:
            manifest_ = json.load(file)
    elif isinstance(config, dict):
        manifest_ = config
    else:
        msg = "config must be a dict or a path"
        raise TypeError(msg)
    io_keys_ = manifest_["_io_keys"]
    class_ = manifest_["class"]
    manifest_.pop("class", None)
    if class_ == "Compute":
        plugin_ = ComputePlugin(_uuid=False, **manifest_)
    elif class_ == "WIPP":
        plugin_ = Plugin(_uuid=False, **manifest_)
    else:
        msg = "Invalid value of class"
        raise ValueError(msg)
    for key, value_ in io_keys_.items():
        val = value_["value"]
        if val is not None:  # exclude those values not set
            setattr(plugin_, key, val)
    return plugin_


def remove_plugin(plugin: str, version: Optional[Union[str, list[str]]] = None) -> None:
    """Remove plugin from the local database."""
    if version is None:
        for plugin_version in PLUGINS[plugin]:
            remove_plugin(plugin, plugin_version)
            return
    else:
        if isinstance(version, list):
            for version_ in version:
                remove_plugin(plugin, version_)
            return
        if not isinstance(version, Version):
            version_ = cast_version(version)
        else:
            version_ = version
        path = PLUGINS[plugin][version_]
        path.unlink()
        refresh()


def remove_all() -> None:
    """Remove all plugins from the local database."""
    organizations = [
        x for x in _PLUGIN_DIR.iterdir() if x.name != "__pycache__" and x.is_dir()
    ]  # ignore __pycache__
    logger.warning("Removing all plugins from local database")
    for org in organizations:
        shutil.rmtree(org)
    refresh()
