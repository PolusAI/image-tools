import enum
import json
import random
import signal
import typing
import re
from copy import deepcopy
from .models.PolusComputeSchema import PluginUIInput, PluginUIOutput
import pathlib
import fsspec
from python_on_whales import docker
import logging
from .models.PolusComputeSchema import PluginSchema as ComputeSchema

logger = logging.getLogger("polus.plugins")


class IOKeyError(Exception):
    pass


class PluginMethods:
    @property
    def organization(self):
        return self.containerId.split("/")[0]

    @property
    def _config_file(self):
        inp = {x.name: str(x.value) for x in self.inputs}
        out = {x.name: str(x.value) for x in self.outputs}
        config = {"inputs": inp, "outputs": out}
        return config

    def save_manifest(self, path: typing.Union[str, pathlib.Path], indent: int = 4):
        with open(path, "w") as fw:
            json.dump(self.manifest, fw, indent=indent)
        logger.debug("Saved manifest to %s" % (path))

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self._config_file, fw)
        logger.debug("Saved config to %s" % (path))

    def load_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "r") as fw:
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
        return json.loads(self.json(exclude={"_io_keys", "versions"}))

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

    def __lt__(self, other):
        return self.version < other.version

    def __gt__(self, other):

        return other.version < self.version

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version={self.version.version})"
