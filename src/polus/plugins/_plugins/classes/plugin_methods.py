"""Methods for all plugin objects."""
import enum
import json
import logging
import pathlib
import random
import signal
import typing
from os.path import relpath

import fsspec
import yaml  # type: ignore
from cwltool.context import RuntimeContext
from cwltool.factory import Factory
from python_on_whales import docker

from polus.plugins._plugins.cwl import CWL_BASE_DICT
from polus.plugins._plugins.io import (
    input_to_cwl,
    io_to_yml,
    output_to_cwl,
    outputs_cwl,
)
from polus.plugins._plugins.utils import name_cleaner

logger = logging.getLogger("polus.plugins")


class IOKeyError(Exception):
    """Raised when trying to set invalid I/O parameter."""


class MissingInputValues(Exception):
    """Raised when there are required input values that have not been set."""


class _PluginMethods:
    def _check_inputs(self):
        """Check if all required inputs have been set."""
        _in = [x for x in self.inputs if x.required and not x.value]  # type: ignore
        if len(_in) > 0:
            raise MissingInputValues(
                f"{[x.name for x in _in]} are required inputs but have not been set"  # type: ignore
            )

    @property
    def organization(self):
        return self.containerId.split("/")[0]

    def load_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path) as fw:
            config = json.load(fw)
        inp = config["inputs"]
        out = config["outputs"]
        for k, v in inp.items():
            if k in self._io_keys:
                setattr(self, k, v)
        for k, v in out.items():
            if k in self._io_keys:
                setattr(self, k, v)
        logger.debug("Loaded config from %s" % (path))

    def run(
        self,
        gpus: typing.Union[None, str, int] = "all",
        **kwargs,
    ):
        self._check_inputs()
        inp_dirs = []
        out_dirs = []

        for i in self.inputs:
            if isinstance(i.value, pathlib.Path):
                inp_dirs.append(str(i.value))

        for o in self.outputs:
            if isinstance(o.value, pathlib.Path):
                out_dirs.append(str(o.value))

        inp_dirs_dict = {x: f"/data/inputs/input{n}" for (n, x) in enumerate(inp_dirs)}
        out_dirs_dict = {
            x: f"/data/outputs/output{n}" for (n, x) in enumerate(out_dirs)
        }

        mnts_in = [
            [f"type=bind,source={k},target={v},readonly"]  # must be a list of lists
            for (k, v) in inp_dirs_dict.items()
        ]
        mnts_out = [
            [f"type=bind,source={k},target={v}"]  # must be a list of lists
            for (k, v) in out_dirs_dict.items()
        ]

        mnts = mnts_in + mnts_out
        args = []

        for i in self.inputs:
            if i.value is not None:  # do not include those with value=None
                i._validate()
                args.append(f"--{i.name}")

                if isinstance(i.value, pathlib.Path):
                    args.append(inp_dirs_dict[str(i.value)])

                elif isinstance(i.value, enum.Enum):
                    args.append(str(i.value._name_))

                else:
                    args.append(str(i.value))

        for o in self.outputs:
            if o.value is not None:  # do not include those with value=None
                o._validate()
                args.append(f"--{o.name}")

                if isinstance(o.value, pathlib.Path):
                    args.append(out_dirs_dict[str(o.value)])

                elif isinstance(o.value, enum.Enum):
                    args.append(str(o.value._name_))

                else:
                    args.append(str(o.value))

        container_name = f"polus{random.randint(10, 99)}"

        def sig(
            signal, frame
        ):  # signal handler to kill container when KeyboardInterrupt
            print(f"Exiting container {container_name}")
            docker.kill(container_name)

        signal.signal(
            signal.SIGINT, sig
        )  # make of sig the handler for KeyboardInterrupt
        if gpus is None:
            logger.info(
                "Running container without GPU. %s version %s"
                % (self.__class__.__name__, self.version.version)
            )
            d = docker.run(
                self.containerId,
                args,
                name=container_name,
                remove=True,
                mounts=mnts,
                **kwargs,
            )
            print(d)
        else:
            logger.info(
                "Running container with GPU: --gpus %s. %s version %s"
                % (gpus, self.__class__.__name__, self.version.version)
            )
            d = docker.run(
                self.containerId,
                args,
                gpus=gpus,
                name=container_name,
                remove=True,
                mounts=mnts,
                **kwargs,
            )
            print(d)

    @property
    def _config(self):
        m = self.dict()
        for x in m["inputs"]:
            x["value"] = None
        return m

    @property
    def manifest(self):
        m = json.loads(self.json(exclude={"_io_keys", "versions"}))
        m["version"] = m["version"]["version"]
        return m

    def __getattribute__(self, name):
        if name != "_io_keys" and hasattr(self, "_io_keys"):
            if name in self._io_keys:
                value = self._io_keys[name].value
                if isinstance(value, enum.Enum):
                    value = value.name
                return value

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "_fs":
            if not issubclass(type(value), fsspec.spec.AbstractFileSystem):
                raise ValueError("_fs must be an fsspec FileSystem")
            else:
                for i in self.inputs:
                    i._fs = value
                for o in self.outputs:
                    o._fs = value
                return

        elif name != "_io_keys" and hasattr(self, "_io_keys"):
            if name in self._io_keys:
                logger.debug(
                    "Value of %s in %s set to %s"
                    % (name, self.__class__.__name__, value)
                )
                self._io_keys[name].value = value
                return
            else:
                raise IOKeyError(
                    "Attempting to set %s in %s but %s is not a valid I/O parameter"
                    % (name, self.__class__.__name__, name)
                )

        super().__setattr__(name, value)

    def _to_cwl(self):
        """Return CWL yml as dict."""
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
        cwl_dict["requirements"]["DockerRequirement"]["dockerPull"] = self.containerId
        return cwl_dict

    def save_cwl(self, path: typing.Union[str, pathlib.Path]):
        """Save plugin as CWL command line tool."""
        assert str(path).split(".")[-1] == "cwl", "Path must end in .cwl"
        with open(path, "w") as file:
            yaml.dump(self._to_cwl(), file)
        return path

    @property
    def _cwl_io(self) -> dict:
        """Dict of I/O for CWL."""
        return {
            x.name: io_to_yml(x) for x in self._io_keys.values() if x.value is not None
        }

    def save_cwl_io(self, path):
        """Save plugin's I/O values to yml file to be used with CWL command line tool."""
        self._check_inputs()
        assert str(path).split(".")[-1] == "yml", "Path must end in .yml"
        with open(path, "w") as file:
            yaml.dump(self._cwl_io, file)
        return path

    def run_cwl(
        self,
        cwl_path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
        io_path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    ):
        """Run configured plugin in CWL.

        Run plugin as a CWL command line tool after setting I/O values.
        Two files will be generated: a CWL (`.cwl`) command line tool
        and an I/O file (`.yml`). They will be generated in
        current working directory if no paths are specified. Optional paths
        for these files can be specified with arguments `cwl_path`,
        and `io_path` respectively.

        Args:
            cwl_path: [Optional] target path for `.cwl` file
            io_path: [Optional] target path for `.yml` file

        """
        if not self.outDir:
            raise ValueError("")

        if not cwl_path:
            _p = pathlib.Path.cwd().joinpath(name_cleaner(self.name) + ".cwl")
            _cwl = self.save_cwl(_p)
        else:
            _cwl = self.save_cwl(cwl_path)

        if not io_path:
            _p = pathlib.Path.cwd().joinpath(name_cleaner(self.name) + ".yml")
            self.save_cwl_io(_p)  # saves io to make it visible to user
        else:
            self.save_cwl_io(io_path)  # saves io to make it visible to user

        outdir_path = relpath(self.outDir.parent)  # type: ignore
        rc = RuntimeContext({"outdir": outdir_path})
        fac = Factory(runtime_context=rc)
        cwl = fac.make(str(_cwl))
        return cwl(**self._cwl_io)  # object's io dict is used instead of .yml file

    def __lt__(self, other):
        return self.version < other.version

    def __gt__(self, other):
        return other.version < self.version

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version={self.version.version})"
