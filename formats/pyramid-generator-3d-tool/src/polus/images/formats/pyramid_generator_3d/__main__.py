"""CLI for the pyramid_generator_3d tool."""

import logging
import os
import pathlib
import typing

import typer
from polus.images.formats.pyramid_generator_3d.pyramid_generator_3d import (
    GroupBy,
    SubCommand,
    gen_py3d,
    gen_volume,
)

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.formats.pyramid_generator_3d")
logger.setLevel(POLUS_LOG)

app = typer.Typer()


def sub_cmd_callback(ctx: typer.Context, value: SubCommand) -> SubCommand:
    """Parse cmd type and set custom context object.

    set ctx.obj["sub_cmd"] according to the subcommand value, such that we can
    check the validity of the parameters for the subcommand.
    Args:
        ctx (typer.Context): typer context object
        value (str): passed in parameter value
    """
    ctx.ensure_object(dict)
    ctx.obj["sub_cmd"] = "Py3D" if value == SubCommand.Py3D else "Vol"
    return value


def _camal_case(s: str) -> str:
    """Convert string to camel case.

    Args:
        s (str): input string

    Returns:
        str: camel case string
    """
    s_ = s.split("_")
    return s_[0] + "".join(word.capitalize() for word in s_[1:])


def vol_option_callback(
    ctx: typer.Context, param: typer.CallbackParam, value: typing.Any
):
    """Determine validity of Vol options if using Vol subcommand.

    Args:
        ctx (typer.Context): typer context object
        value (str): passed in parameter value
    """
    if ctx.obj["sub_cmd"].startswith("Vol"):
        if not value:
            raise typer.BadParameter(
                f"--{_camal_case(param.name)} are required for volume generation."
            )
    if param.name == "group_by":
        ctx.obj["group_by"] = value
    return value


def inp_dir_callback(ctx: typer.Context, value: typing.Union[pathlib.Path, None]):
    """Determine validity of input directory if using Vol subcommand.

    Args:
        ctx (typer.Context): typer context object
        value (pathlib.Path): passed in parameter value
    """
    if ctx.obj["sub_cmd"].startswith("Vol"):
        if not value:
            raise typer.BadParameter(
                "Input directory is required for volume generation."
            )
        if not value.exists():
            raise typer.BadParameter("Input directory does not exist.")
        if not value.is_dir():
            raise typer.BadParameter("Input directory is not a directory.")
        if not os.access(value, os.R_OK):
            raise typer.BadParameter("Input directory is not readable.")

    return value.resolve() if value else value


def py3d_option_callback(
    ctx: typer.Context, param: typer.CallbackParam, value: typing.Any
):
    """Determine validity of parameter if using Py3D subcommand.

    Args:
        ctx (typer.Context): typer context object
        value (int): passed in parameter value
    """
    if ctx.obj["sub_cmd"] == "Py3D":
        if not value:
            raise typer.BadParameter(
                f"--{_camal_case(param.name)} is required for 3D pyramid generation."
            )
    return value


def zarr_dir_callback(ctx: typer.Context, value: typing.Union[pathlib.Path, None]):
    """Determine validity of zarr directory if using Py3D subcommand.

    Args:
        ctx (typer.Context): typer context object
        value (pathlib.Path): passed in parameter value
    """
    if ctx.obj["sub_cmd"] == "Py3D":
        if value:  # use zarr directory
            if not value.exists():
                raise typer.BadParameter("Zarr directory does not exist.")
            if not value.is_dir():
                raise typer.BadParameter("Zarr directory is not a directory.")
            if not os.access(value, os.R_OK):
                raise typer.BadParameter("Zarr directory is not readable.")
        else:  # None value, use inpDir instead
            # change context label
            ctx.obj["sub_cmd"] = "Vol_Py3D"
            logger.info(
                "Zarr dir not provided, using inpDir to perform volume "
                "generation first, and then 3D pyramid generation."
            )

    # fully resolve path
    return value.resolve() if value else value


@app.command()
def main(
    ctx: typer.Context,
    sub_cmd: SubCommand = typer.Option(
        ...,
        "--subCmd",
        help="Subcommand to run. Choose 'Py3D' for 3D pyramid generation or 'Vol' for volume generation.",
        callback=sub_cmd_callback,
    ),
    zarr_dir: pathlib.Path = typer.Option(
        None,
        "--zarrDir",
        help=(
            "Directory containing the input zarr files for 3D pyramid generation. "
            "If not provided, inpDir needs to be provided."
        ),
        callback=zarr_dir_callback,
    ),
    inp_dir: pathlib.Path = typer.Option(
        None,
        "--inpDir",
        help="Directory containing the input images for 3D pyramid generation.",
        callback=inp_dir_callback,
    ),
    file_pattern: str = typer.Option(
        None,
        "--filePattern",
        help="File pattern for selecting images for 3D pyramid generation.",
        callback=vol_option_callback,
    ),
    group_by: GroupBy = typer.Option(
        None,
        "--groupBy",
        help="Image dimension, e.g., 'z', to group images for 3D pyramid generation.",
        callback=vol_option_callback,
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory.",
        exists=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    out_img_name: str = typer.Option(
        None,
        "--outImgName",
        help="Name of the output image name for 3D pyramid generation.",
        callback=vol_option_callback,
    ),
    base_scale_key: int = typer.Option(
        0,
        "--baseScaleKey",
        help="Base scale key for volume generation.",
    ),
    num_levels: int = typer.Option(
        None,
        "--numLevels",
        help="Number of levels for volume generation.",
        callback=py3d_option_callback,
    ),
):
    """CLI for the pyramid_generator_3d tool."""
    # for some reason after the callback the sub_cmd only receives None value.
    # need to determine type based on stored context custom object. nevertheless
    # other param checks are correct based on the sub_cmd value
    # seems to be problem specific to using typer callback and enum
    sub_cmd = ctx.obj["sub_cmd"]
    group_by = ctx.obj.get("group_by", None)
    logger.info("Starting pyramid_generator_3d...")
    logger.info("subCmd: %s", sub_cmd)
    logger.info("zarrDir: %s", zarr_dir)
    logger.info("inpDir: %s", inp_dir)
    logger.info("groupBy: %s", group_by)
    logger.info("filePattern: %s", file_pattern)
    logger.info("outDir: %s", out_dir)
    logger.info("outImgName: %s", out_img_name)
    logger.info("baseScaleKey: %d", base_scale_key)
    logger.info("numLevels: %s", num_levels)

    # call argolid
    if sub_cmd.startswith("Vol"):
        logger.info("Starting Volume Generation...")
        gen_volume(inp_dir, group_by, file_pattern, out_dir, out_img_name)
        logger.info("Volume generation completed.")

    if sub_cmd.endswith("Py3D"):
        logger.info("Starting 3D Pyramid Generation...")
        # use out_dir if zarr_dir is None. volume generation outputs zarr to out_dir
        zarr_dir = zarr_dir if zarr_dir else out_dir / out_img_name
        gen_py3d(zarr_dir, base_scale_key, num_levels)
        logger.info("3D pyramid generation completed.")


if __name__ == "__main__":
    app()
