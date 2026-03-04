"""Nyxus Plugin."""
import logging
import pathlib
from typing import Any
from typing import Optional
from typing import Union

import vaex
from nyxus import Nyxus

from .utils import Extension

logger = logging.getLogger(__name__)

chunk_size = 100_000


def nyxus_func(  # noqa: PLR0913
    int_file: Union[list[pathlib.Path], Any],
    seg_file: Union[list[pathlib.Path], Any],
    out_dir: pathlib.Path,
    features: list[str],
    file_extension: Extension,
    pixels_per_micron: Optional[float] = 1.0,
    neighbor_dist: Optional[int] = 5,
    single_roi: Optional[bool] = False,
) -> None:
    """Scalable Extraction of Nyxus Features.

    Args:
        int_file : Path to intensity image(s).
        seg_file : Path to label image.
        out_dir : Path to output directory.
        features : List of features to compute.
        file_extension: Output file extension.
        pixels_per_micron : Number of pixels for every micrometer.
        neighbor_dist : Pixel distance between neighbor objects. Defaults to 5.
        single_roi : 'True' to treat intensity image as single roi and vice versa.
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

    if hasattr(file_extension, "value"):
        file_ext = str(file_extension.value).lower().strip()
    else:
        file_ext = str(file_extension).lower().strip()

    ext_map = {
        "arrowipc": ".arrow",
        "parquet": ".parquet",
        "pandas": ".csv",
    }

    try:
        ext = ext_map[file_ext]
    except KeyError:
        raise ValueError(
            f"Unsupported file_extension '{file_extension}'. "
            f"Must be one of {list(ext_map)}"
        )

    for i_file in int_file:
        out_name = i_file.name.replace("".join(i_file.suffixes), ext)
        output_path = str(pathlib.Path(out_dir, out_name))

        feats = nyx.featurize_files(
            intensity_files=[str(i_file)],
            mask_files=[str(seg_file[0])],
            single_roi=single_roi,
            output_type=file_ext,
            output_path=output_path,
        )

        if file_ext == "pandas":
            vf = vaex.from_pandas(feats)
            vf.export_csv(path=output_path, chunk_size=chunk_size)
