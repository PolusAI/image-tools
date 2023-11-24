"""Methods for all plugin objects."""
# pylint: disable=W1203, W0212, enable=W1201
import enum
import json
import logging
import random
import signal
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TypeVar
from typing import Union

import fsspec
import yaml  # type: ignore
from cwltool.context import RuntimeContext
from cwltool.factory import Factory
from cwltool.utils import CWLObjectType
from polus.plugins._plugins.cwl import CWL_BASE_DICT
from polus.plugins._plugins.io import input_to_cwl
from polus.plugins._plugins.io import io_to_yml
from polus.plugins._plugins.io import output_to_cwl
from polus.plugins._plugins.io import outputs_cwl
from polus.plugins._plugins.utils import name_cleaner
from python_on_whales import docker

logger = logging.getLogger("polus.plugins")

StrPath = TypeVar("StrPath", str, Path)


class IOKeyError(Exception):
    """Raised when trying to set invalid I/O parameter."""


class MissingInputValuesError(Exception):
    """Raised when there are required input values that have not been set."""


class BasePlugin:
    """Base Class for Plugins."""

    def _check_inputs(self) -> None:
        """Check if all required inputs have been set."""
        _in = [x for x in self.inputs if x.required and not x.value]  # type: ignore
        if len(_in) > 0:
            msg = f"{[x.name for x in _in]} are required inputs but have not been set"
            raise MissingInputValuesError(
                msg,  # type: ignore
            )

    @property
    def organization(self) -> str:
        """Plugin container's organization."""
        return self.containerId.split("/")[0]

    def load_config(self, path: StrPath) -> None:
        """Load configured plugin from file."""
        with Path(path).open(encoding="utf=8") as fw:
            config = json.load(fw)
        inp = config["inputs"]
        out = config["outputs"]
        for k, v in inp.items():
            if k in self._io_keys:
                setattr(self, k, v)
        for k, v in out.items():
            if k in self._io_keys:
                setattr(self, k, v)
        logger.debug(f"Loaded config from {path}")

    def run(
        self,
        gpus: Union[None, str, int] = "all",
        **kwargs: Union[None, str, int],
    ) -> None:
        """Run plugin in Docker container."""
        self._check_inputs()
        inp_dirs = [x for x in self.inputs if isinstance(x.value, Path)]
        out_dirs = [x for x in self.outputs if isinstance(x.value, Path)]

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

                if isinstance(i.value, Path):
                    args.append(inp_dirs_dict[str(i.value)])

                elif isinstance(i.value, enum.Enum):
                    args.append(str(i.value._name_))

                else:
                    args.append(str(i.value))

        for o in self.outputs:
            if o.value is not None:  # do not include those with value=None
                o._validate()
                args.append(f"--{o.name}")

                if isinstance(o.value, Path):
                    args.append(out_dirs_dict[str(o.value)])

                elif isinstance(o.value, enum.Enum):
                    args.append(str(o.value._name_))

                else:
                    args.append(str(o.value))

        random_int = random.randint(10, 99)  # noqa: S311 # only for naming
        container_name = f"polus{random_int}"

        def sig(
            signal,  # noqa # pylint: disable=W0613, W0621
            frame,  # noqa # pylint: disable=W0613, W0621
        ) -> None:  # signal handler to kill container when KeyboardInterrupt
            logger.info(f"Exiting container {container_name}")
            docker.kill(container_name)

        signal.signal(
            signal.SIGINT,
            sig,
        )  # make of sig the handler for KeyboardInterrupt
        if gpus is None:
            logger.info(
                f"""Running container without GPU. {self.__class__.__name__}
                version {self.version!s}""",
            )
            docker_ = docker.run(
                self.containerId,
                args,
                name=container_name,
                remove=True,
                mounts=mnts,
                **kwargs,
            )
            print(docker_)  # noqa
        else:
            logger.info(
                f"""Running container with GPU: --gpus {gpus}.
                {self.__class__.__name__} version {self.version!s}""",
            )
            docker_ = docker.run(
                self.containerId,
                args,
                gpus=gpus,
                name=container_name,
                remove=True,
                mounts=mnts,
                **kwargs,
            )
            print(docker_)  # noqa

    @property
    def _config(self) -> dict:
        model_ = self.dict()
        for inp in model_["inputs"]:
            inp["value"] = None
        return model_

    @property
    def manifest(self) -> dict:
        """Plugin manifest."""
        manifest_ = json.loads(self.json(exclude={"_io_keys", "versions", "id"}))
        manifest_["version"] = manifest_["version"]["version"]
        return manifest_

    def __getattribute__(self, name: str) -> Any:  # noqa
        if name == "__class__":  # pydantic v2 change
            return super().__getattribute__(name)
        if name != "_io_keys" and hasattr(self, "_io_keys") and name in self._io_keys:
            value = self._io_keys[name].value
            if isinstance(value, enum.Enum):
                value = value.name
            return value

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa
        if name == "_fs":
            if not issubclass(type(value), fsspec.spec.AbstractFileSystem):
                msg = "_fs must be an fsspec FileSystem"
                raise ValueError(msg)
            for i in self.inputs:
                i._fs = value
            for o in self.outputs:
                o._fs = value
            return

        if name != "_io_keys" and hasattr(self, "_io_keys"):
            if name in self._io_keys:
                logger.debug(
                    f"Value of {name} in {self.__class__.__name__} set to {value}",
                )
                self._io_keys[name].value = value
                return
            msg = (
                f"Attempting to set {name} in "
                "{self.__class__.__name__} but "
                "{{name}} is not a valid I/O parameter"
            )
            raise IOKeyError(
                msg,
            )

        super().__setattr__(name, value)

    def _to_cwl(self) -> dict:
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

    def save_cwl(self, path: StrPath) -> Path:
        """Save plugin as CWL command line tool."""
        if str(path).rsplit(".", maxsplit=1)[-1] != "cwl":
            msg = "path must end in .cwl"
            raise ValueError(msg)
        with Path(path).open("w", encoding="utf-8") as file:
            yaml.dump(self._to_cwl(), file)
        return Path(path)

    @property
    def _cwl_io(self) -> dict:
        """Dict of I/O for CWL."""
        return {
            x.name: io_to_yml(x) for x in self._io_keys.values() if x.value is not None
        }

    def save_cwl_io(self, path: StrPath) -> Path:
        """Save plugin's I/O values to yml file.

        To be used with CWL Command Line Tool.
        """
        self._check_inputs()
        if str(path).rsplit(".", maxsplit=1)[-1] != "yml":
            msg = "path must end in .yml"
            raise ValueError(msg)
        with Path(path).open("w", encoding="utf-8") as file:
            yaml.dump(self._cwl_io, file)
        return Path(path)

    def run_cwl(
        self,
        cwl_path: Optional[StrPath] = None,
        io_path: Optional[StrPath] = None,
    ) -> Union[CWLObjectType, str, None]:
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
            msg = ""
            raise ValueError(msg)

        if not cwl_path:
            _p = Path.cwd().joinpath(name_cleaner(self.name) + ".cwl")
            _cwl = self.save_cwl(_p)
        else:
            _cwl = self.save_cwl(cwl_path)

        if not io_path:
            _p = Path.cwd().joinpath(name_cleaner(self.name) + ".yml")
            self.save_cwl_io(_p)  # saves io to make it visible to user
        else:
            self.save_cwl_io(io_path)  # saves io to make it visible to user

        outdir_path = self.outDir.parent.relative_to(Path.cwd())
        r_c = RuntimeContext({"outdir": str(outdir_path)})
        fac = Factory(runtime_context=r_c)
        cwl = fac.make(str(_cwl))
        return cwl(**self._cwl_io)  # object's io dict is used instead of .yml file

    def __lt__(self, other: "BasePlugin") -> bool:
        return self.version < other.version

    def __gt__(self, other: "BasePlugin") -> bool:
        return other.version < self.version

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', version={self.version!s})"
        )
