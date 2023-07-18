"""Nyxus Plugin."""
import logging
import pathlib
from typing import Any, List, Optional, Union

import vaex
from nyxus import Nyxus

from polus.plugins.features.nyxus_plugin.utils import Extension

logger = logging.getLogger(__name__)

chunk_size = 100_000


def nyxus_func(
    int_file: Union[List[pathlib.Path], Any],
    seg_file: Union[List[pathlib.Path], Any],
    out_dir: pathlib.Path,
    features: List[str],
    file_extension: Extension,
    pixels_per_micron: Optional[float] = 1.0,
    neighbor_dist: Optional[int] = 5,
) -> None:
    """Scalable Extraction of Nyxus Features.

    Args:
        int_file : Path to intensity image(s).
        seg_file : Path to label image.
        out_dir : Path to output directory.
        features : Pattern to parse image replicates.
        file_extension: Output file extension.
        pixels_per_micron : Number of pixels for every micrometer.
        neighbor_dist : Pixel distance between neighbor objects. Defaults to 5.0.
    """
    if isinstance(int_file, pathlib.Path):
        int_file = [int_file]

    nyx = Nyxus(features)

    nyx_params = {
        "neighbor_distance": neighbor_dist,
        "pixels_per_micron": pixels_per_micron,
        "n_feature_calc_threads": 4,
    }

    nyx.set_params(**nyx_params)

    for i_file in int_file:
        feats = nyx.featurize_files(
            intensity_files=[str(i_file)],
            mask_files=[str(seg_file[0])],
        )

    out_name = i_file.name.replace("".join(i_file.suffixes), f"{file_extension}")
    out_name = pathlib.Path(out_dir, out_name)
    vf = vaex.from_pandas(feats)
    if f"{file_extension}" in [".feather", ".arrow"]:
        vf.export_feather(out_name)
    else:
        vf.export_csv(path=out_name, chunk_size=chunk_size)
