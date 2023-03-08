"""Classes for Plugin objects containing methods to configure, run, and save."""
import json
import logging
import pathlib
import typing
import uuid
from copy import deepcopy

from pydantic import Extra

from polus.plugins._plugins.classes.plugin_methods import _PluginMethods
from polus.plugins._plugins.io import (
    DuplicateVersionFound,
    Version,
    _in_old_to_new,
    _ui_old_to_new,
)
from polus.plugins._plugins.manifests.manifest_utils import (
    InvalidManifest,
    _load_manifest,
    validate_manifest,
)
from polus.plugins._plugins.models import (
    ComputeSchema,
    PluginUIInput,
    PluginUIOutput,
    WIPPPluginManifest,
)
from polus.plugins._plugins.utils import cast_version, name_cleaner

logger = logging.getLogger("polus.plugins")
PLUGINS: typing.Dict[str, typing.Dict] = {}
# PLUGINS = {"BasicFlatfieldCorrectionPlugin":
#               {Version('0.1.4'): Path(<...>), Version('0.1.5'): Path(<...>)}.
#            "VectorToLabel": {Version(...)}}

"""
Paths and Fields
"""
# Location to store any discovered plugin manifests
PLUGIN_DIR = pathlib.Path(__file__).parent.parent.joinpath("manifests")


def load_config(config: typing.Union[dict, pathlib.Path]):
    """Load configured plugin from config file/dict."""
    if isinstance(config, pathlib.Path):
        with open(config) as fr:
            m = json.load(fr)
    elif isinstance(config, dict):
        m = config
    else:
        raise TypeError("config must be a dict or a path")
    _io = m["_io_keys"]
    cl = m["class"]
    m.pop("class", None)
    if cl == "Compute":
        pl = ComputePlugin(_uuid=False, **m)
    elif cl == "WIPP":
        pl = Plugin(_uuid=False, **m)
    else:
        raise ValueError("Invalid value of class")
    for k, v in _io.items():
        val = v["value"]
        if val is not None:  # exclude those values not set
            setattr(pl, k, val)
    return pl


def get_plugin(name: str, version: typing.Optional[str] = None):
    """Get a plugin with option to specify version.

    Return a plugin object with the option to specify a version. The specified version's manifest must exist in manifests folder.

    Args:
        name: Name of the plugin.
        version: Optional version of the plugin, must follow semver.

    Returns:
        Plugin object
    """
    if version is None:
        return load_plugin(PLUGINS[name][max(PLUGINS[name])])
    else:
        return load_plugin(PLUGINS[name][Version(**{"version": version})])


def refresh():
    """Refresh the plugin list."""
    organizations = [
        x for x in PLUGIN_DIR.iterdir() if x.name != "__pycache__"
    ]  # ignore __pycache__

    global _invalid
    _invalid = {}

    for org in organizations:
        if org.is_file():
            continue

        for file in org.iterdir():
            if file.suffix == ".py":
                continue

            try:
                plugin = validate_manifest(file)
            except InvalidManifest as e:
                _invalid[_load_manifest(file)["name"]] = str(e.__cause__)
            else:
                key = name_cleaner(plugin.name)
                # Add version and path to VERSIONS
                if key not in PLUGINS:
                    PLUGINS[key] = {}
                if plugin.version in PLUGINS[key]:
                    if not file == PLUGINS[key][plugin.version]:
                        raise DuplicateVersionFound(
                            "Found duplicate version of plugin %s in %s"
                            % (plugin.name, PLUGIN_DIR)
                        )
                PLUGINS[key][plugin.version] = file

    if len(_invalid) > 0:
        logger.warning(
            f"local manifests {[str(x) for x in _invalid.keys()]} are invalid. Run polus.plugins.print_invalid() for more details."
        )


def print_invalid():
    """Print invalid manifests with respective validation errors."""
    for x, y in _invalid.items():
        print(x)
        print(y + "\n")


_r = refresh


def list_plugins():
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

    id: uuid.UUID

    class Config:
        """Config class for Pydantic Model."""

        extra = Extra.allow
        allow_mutation = False

    def __init__(self, _uuid: bool = True, **data):
        """Init a plugin object from manifest."""
        if _uuid:
            data["id"] = uuid.uuid4()
        else:
            data["id"] = uuid.UUID(str(data["id"]))

        data["version"] = cast_version(data["version"])
        super().__init__(**data)

        self.Config.allow_mutation = True
        self._io_keys = {i.name: i for i in self.inputs}
        self._io_keys.update({o.name: o for o in self.outputs})

        if self.author == "":
            logger.warning(
                f"The plugin ({self.name}) is missing the author field. This field is not required but should be filled in."
            )

    @property
    def versions(plugin):  # cannot be in PluginMethods because PLUGINS lives here
        """Return list of local versions of a Plugin."""
        return list(PLUGINS[name_cleaner(plugin.name)])

    def to_compute(self, hardware_requirements: typing.Optional[dict] = None):
        """Convert WIPP Plugin object to Compute Plugin object."""
        data = deepcopy(self.manifest)
        return ComputePlugin(
            hardware_requirements=hardware_requirements, _from_old=True, **data
        )

    def save_manifest(
        self,
        path: typing.Union[str, pathlib.Path],
        hardware_requirements: typing.Optional[dict] = None,
        compute: bool = False,
    ):
        """Save plugin manifest to specified path."""
        if compute:
            with open(path, "w") as fw:
                self.to_compute(
                    hardware_requirements=hardware_requirements
                ).save_manifest(path)
        else:
            with open(path, "w") as fw:
                d = self.manifest
                json.dump(
                    d,
                    fw,
                    indent=4,
                )

        logger.debug("Saved manifest to %s" % (path))

    def __setattr__(self, name, value):
        """Set I/O parameters as attributes."""
        _PluginMethods.__setattr__(self, name, value)

    @property
    def _config_file(self):
        m = self._config
        m["class"] = "WIPP"
        return m

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        """Save manifest with configured I/O parameters to specified path."""
        with open(path, "w") as fw:
            json.dump(self._config_file, fw, indent=4, default=str)
        logger.debug("Saved config to %s" % (path))

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
        hardware_requirements: typing.Optional[dict] = None,
        _from_old: bool = False,
        _uuid: bool = True,
        **data,
    ):
        """Init a plugin object from manifest."""
        if _uuid:
            data["id"] = uuid.uuid4()
        else:
            data["id"] = uuid.UUID(str(data["id"]))

        if _from_old:

            def _convert_input(d: dict):
                d["type"] = _in_old_to_new(d["type"])
                return d

            def _convert_output(d: dict):
                d["type"] = "path"
                return d

            def _ui_in(d: dict):  # assuming old all ui input
                # assuming format inputs. ___
                inp = d["key"].split(".")[-1]  # e.g inpDir
                try:
                    tp = [x["type"] for x in data["inputs"] if x["name"] == inp][
                        0
                    ]  # get type from i/o
                except IndexError:
                    tp = "string"  # default to string
                except BaseException:
                    raise

                d["type"] = _ui_old_to_new(tp)
                return PluginUIInput(**d)

            def _ui_out(d: dict):
                nd = deepcopy(d)
                nd["name"] = "outputs." + nd["name"]
                nd["type"] = _ui_old_to_new(nd["type"])
                return PluginUIOutput(**nd)

            data["inputs"] = [_convert_input(x) for x in data["inputs"]]
            data["outputs"] = [_convert_output(x) for x in data["outputs"]]
            data["pluginHardwareRequirements"] = {}
            data["ui"] = [_ui_in(x) for x in data["ui"]]  # inputs
            data["ui"].extend([_ui_out(x) for x in data["outputs"]])  # type: ignore # outputs

        if hardware_requirements:
            for k, v in hardware_requirements.items():
                data["pluginHardwareRequirements"][k] = v

        data["version"] = cast_version(data["version"])
        super().__init__(**data)
        self.Config.allow_mutation = True
        self._io_keys = {i.name: i for i in self.inputs}
        self._io_keys.update({o.name: o for o in self.outputs})  # type: ignore

        if self.author == "":
            logger.warning(
                f"The plugin ({self.name}) is missing the author field. This field is not required but should be filled in."
            )

    @property
    def versions(plugin):  # cannot be in PluginMethods because PLUGINS lives here
        """Return list of local versions of a Plugin."""
        return list(PLUGINS[name_cleaner(plugin.name)])

    @property
    def _config_file(self):
        m = self._config
        m["class"] = "Compute"
        return m

    def __setattr__(self, name, value):
        """Set I/O parameters as attributes."""
        _PluginMethods.__setattr__(self, name, value)

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        """Save configured manifest with I/O parameters to specified path."""
        with open(path, "w") as fw:
            json.dump(self._config_file, fw, indent=4)
        logger.debug("Saved config to %s" % (path))

    def save_manifest(self, path: typing.Union[str, pathlib.Path]):
        """Save plugin manifest to specified path."""
        with open(path, "w") as fw:
            json.dump(self.manifest, fw, indent=4)
        logger.debug("Saved manifest to %s" % (path))

    def __repr__(self) -> str:
        """Print plugin name and version."""
        return _PluginMethods.__repr__(self)


def load_plugin(
    manifest: typing.Union[str, dict, pathlib.Path]
) -> typing.Union[Plugin, ComputePlugin]:
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
    manifest: typing.Union[str, dict, pathlib.Path],
    refresh: bool = False,
):
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
    org_path = PLUGIN_DIR.joinpath(organization.lower())
    org_path.mkdir(exist_ok=True, parents=True)
    if not org_path.joinpath(out_name).exists():
        with open(org_path.joinpath(out_name), "w") as fw:
            json.dump(plugin.dict(), fw, indent=4)

    # Refresh plugins list if refresh = True
    if refresh:
        _r()
    return plugin
