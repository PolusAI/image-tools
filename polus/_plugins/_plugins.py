from copy import deepcopy
from pprint import pformat
import typing
from ._io import Version, DuplicateVersionFound
from ._plugin_model import WIPPPluginManifest
from ._utils import (
    name_cleaner,
    input_to_cwl,
    output_to_cwl,
    outputs_cwl,
    io_to_yml,
    cast_version,
)
from ._plugin_methods import PluginMethods
from .PolusComputeSchema import PluginUIInput, PluginUIOutput
from .PolusComputeSchema import PluginSchema as ComputeSchema
from ._manifests import _load_manifest, validate_manifest
from ._io import Version, DuplicateVersionFound, _in_old_to_new, _ui_old_to_new
from ._cwl import CWL_BASE_DICT
from pydantic import Extra
import pathlib
import json
import uuid
import logging
import yaml

logger = logging.getLogger("polus.plugins")
PLUGINS = {}
# PLUGINS = {"BasicFlatfieldCorrectionPlugin":
#               {Version('0.1.4'): Path(<...>), Version('0.1.5'): Path(<...>)}.
#            "VectorToLabel": {Version(...)}}

"""
Paths and Fields
"""
# Location to store any discovered plugin manifests
PLUGIN_DIR = pathlib.Path(__file__).parent.parent.joinpath("manifests")


"""
Plugin Fetcher Class
"""


class _Plugins:
    def __getattribute__(self, name):
        if name in PLUGINS:
            return self.get_plugin(name)
        return super().__getattribute__(name)

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return pformat(self.list)

    @property
    def list(self):
        output = list(PLUGINS.keys())
        output.sort()
        return output

    @classmethod
    def get_plugin(cls, name: str, version: typing.Optional[str] = None):
        """Returns a plugin object.

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

    def load_config(self, config: typing.Union[dict, pathlib.Path]):
        if isinstance(config, pathlib.Path):
            with open(config, "r") as fr:
                m = json.load(fr)
        elif isinstance(config, dict):
            m = config
        else:
            raise TypeError("config must be a dict or a path")
        _io = m["_io_keys"]
        cl = m["class"]
        m.pop("class", None)
        if cl == "NewPlugin":
            pl = ComputePlugin(_uuid=False, **m)
        elif cl == "OldPlugin":
            pl = Plugin(_uuid=False, **m)
        else:
            raise ValueError("Invalid value of class")
        for k, v in _io.items():
            val = v["value"]
            if val is not None:  # exclude those values not set
                setattr(pl, k, val)
        return pl

    @classmethod
    def refresh(cls):
        """Refresh the plugin list

        This should be optimized, since it will become noticeably slow when
        there are many plugins.
        """

        organizations = list(PLUGIN_DIR.iterdir())

        for org in organizations:

            if org.is_file():
                continue

            for file in org.iterdir():

                plugin = validate_manifest(pathlib.Path(file))
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


class Plugin(WIPPPluginManifest, PluginMethods):
    """Required until json schema is fixed"""

    id: uuid.UUID

    class Config:
        extra = Extra.allow
        allow_mutation = False

    def __init__(self, _uuid: bool = True, **data):

        if _uuid:
            data["id"] = uuid.uuid4()
        else:
            data["id"] = uuid.UUID(str(data["id"]))

        data["version"] = cast_version(data["version"])  # cast version

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
        """Return list of versions of a Plugin"""
        return list(PLUGINS[name_cleaner(plugin.name)])

    def new_schema(self, hardware_requirements: typing.Optional[dict] = None):
        data = deepcopy(self.manifest)
        return ComputePlugin(
            hardware_requirements=hardware_requirements, _from_old=True, **data
        )

    def save_manifest(
        self,
        path: typing.Union[str, pathlib.Path],
        hardware_requirements: typing.Optional[dict] = None,
        new: bool = False,
    ):
        if new:
            with open(path, "w") as fw:
                self.new_schema(
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
        PluginMethods.__setattr__(self, name, value)

    @property
    def _config_file(self):
        m = self._config
        m["class"] = "OldPlugin"
        return m

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self._config_file, fw, indent=4)
        logger.debug("Saved config to %s" % (path))

    def __repr__(self) -> str:
        return PluginMethods.__repr__(self)


class ComputePlugin(ComputeSchema, PluginMethods):
    class Config:
        extra = Extra.allow
        allow_mutation = False

    def __init__(
        self,
        hardware_requirements: typing.Optional[dict] = None,
        _from_old: bool = False,
        _uuid: bool = True,
        **data,
    ):

        if _uuid:
            data["id"] = uuid.uuid4()
        else:
            data["id"] = uuid.UUID(str(data["id"]))

        data["version"] = cast_version(data["version"])  # cast version

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
            data["ui"].extend([_ui_out(x) for x in data["outputs"]])  # outputs

        if hardware_requirements:
            for k, v in hardware_requirements.items():
                data["pluginHardwareRequirements"][k] = v
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
        """Return list of versions of a Plugin"""
        return list(PLUGINS[name_cleaner(plugin.name)])

    @property
    def _config_file(self):
        m = self._config
        m["class"] = "NewPlugin"
        return m

    def __setattr__(self, name, value):
        PluginMethods.__setattr__(self, name, value)

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self._config_file, fw, indent=4)
        logger.debug("Saved config to %s" % (path))

    def save_manifest(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self.manifest, fw, indent=4)
        logger.debug("Saved manifest to %s" % (path))

    def _to_cwl(self):
        cwl_dict = CWL_BASE_DICT
        cwl_dict["inputs"] = {}
        cwl_dict["outputs"] = {}
        inputs = [input_to_cwl(x) for x in self.inputs]
        inputs = inputs + [output_to_cwl(x) for x in self.outputs]
        for inp in inputs:
            cwl_dict["inputs"].update(inp)
        outputs = [outputs_cwl(x) for x in self.outputs]
        for out in outputs:
            cwl_dict["outputs"].update(out)
        cwl_dict["hints"]["DockerRequirement"]["dockerPull"] = self.containerId
        return cwl_dict

    def save_cwl(self, path: typing.Union[str, pathlib.Path]):
        assert str(path).split(".")[-1] == "cwl", "Path must end in .cwl"
        with open(path, "w") as file:
            yaml.dump(self._to_cwl(), file)
        return path

    def _cwl_io(self):
        return {
            x.name: io_to_yml(x) for x in self._io_keys.values() if x.value is not None
        }

    def save_cwl_io(self, path):
        assert str(path).split(".")[-1] == "yml", "Path must end in .yml"
        with open(path, "w") as file:
            yaml.dump(self._cwl_io(), file)
        return path

    def __repr__(self) -> str:
        return PluginMethods.__repr__(self)


def load_plugin(
    manifest: typing.Union[str, dict, pathlib.Path]
) -> typing.Union[Plugin, ComputePlugin]:
    """Parses a manifest and returns one of Plugin or ComputePlugin"""
    manifest = _load_manifest(manifest)
    if "pluginHardwareRequirements" in manifest:
        # Parse the manifest
        plugin = ComputePlugin(**manifest)  # New Schema
    else:
        # Parse the manifest
        plugin = Plugin(**manifest)  # Old Schema
    return plugin


def submit_plugin(
    manifest: typing.Union[str, dict, pathlib.Path],
    refresh: bool = False,
):
    """Parses a plugin and creates a local copy of it.

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
        _Plugins.refresh()
    return plugin
