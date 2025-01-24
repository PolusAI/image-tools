"""pyramid_generator_3d."""
from enum import Enum
from pathlib import Path

from argolid import PyramidGenerator3D, VolumeGenerator


class SubCommand(str, Enum):
    """SubCommand."""

    Py3D = "Py3D"  # only perform 3D pyramid generation
    Vol = "Vol"  # only perform volume generation


def gen_volume(
    inp_dir: Path,
    group_by: str,
    file_pattern: str,
    out_dir: Path,
    out_img_name: str,
):
    """Generate volume.using argolid.

    Args:
        inp_dir (Path): input directory
        group_by (str): image dimension to group by
        file_pattern (str): file pattern to search for images
        out_dir (Path): output directory
        out_img_name (str): output image name
    """
    volume_gen = VolumeGenerator(
        str(inp_dir), group_by, file_pattern, str(out_dir), out_img_name
    )
    volume_gen.generate_volume()


def gen_py3d(
    zarr_dir: Path,
    base_scale_key: int,
    num_levels: int,
):
    """Generate 3d pyramid using argolid.

    Args:
        zarr_dir (Path): path to zarr arrays
        base_scale_key (int): base scale key
        num_levels (int): number of levels for pyramid
    """
    pyramid_gen = PyramidGenerator3D(zarr_dir, base_scale_key)
    pyramid_gen.generate_pyramid(num_levels)
