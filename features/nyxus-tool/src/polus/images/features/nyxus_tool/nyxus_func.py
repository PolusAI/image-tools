"""Nyxus Plugin."""
import logging
import pathlib
from typing import Any

import numpy as np
from bfio import BioReader
from nyxus import Nyxus

from .utils import Extension

logger = logging.getLogger(__name__)

chunk_size = 100_000


# Map: extension -> (nyxus_output_type, file_suffix)
_EXT_MAP: dict[str, tuple[str, str]] = {
    ".csv": ("pandas", ".csv"),
    ".arrow": ("arrowipc", ".arrow"),
    ".parquet": ("parquet", ".parquet"),
}


def _resolve_file_ext(file_extension: Extension | str) -> tuple[str, str]:
    """Resolve extension into (nyxus_output_type, file_suffix)."""
    if hasattr(file_extension, "value"):
        ext = str(file_extension.value).lower().strip()
    else:
        ext = str(file_extension).lower().strip()

    if ext not in _EXT_MAP:
        msg = f"Invalid extension '{file_extension}'. Options: {list(_EXT_MAP)}"
        raise ValueError(
            msg,
        )

    return _EXT_MAP[ext]


def run_nyxus_object_features(  # noqa: PLR0913
    int_file: list[pathlib.Path] | Any,
    seg_file: list[pathlib.Path] | Any,
    out_dir: pathlib.Path,
    features: list[str],
    file_extension: Extension,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Scalable Extraction of Nyxus Features.

    Args:
        int_file : Path to intensity image(s).
        seg_file : Path to label image.
        out_dir : Path to output directory.
        features : List of features to compute.
        file_extension: Output file format (Extension enum or string).
        kwargs: Additional parameters passed to Nyxus.set_params().
    """
    if isinstance(int_file, pathlib.Path):
        int_file = [int_file]

    if isinstance(seg_file, pathlib.Path):
        seg_file = [seg_file]

    nyx = Nyxus(features)

    if kwargs is None:
        kwargs = {}

    nyx.set_params(**kwargs)

    file_ext, suffix = _resolve_file_ext(file_extension)

    for i_file in int_file:
        out_name = i_file.name.replace("".join(i_file.suffixes), suffix)
        output_path = pathlib.Path(out_dir, out_name)

        # Nyxus write the file directly for all formats:
        # - "pandas"   -> writes .csv
        # - "arrowipc" -> writes .arrow
        # - "parquet"  -> writes .parquet
        if file_ext != "pandas":
            nyx.featurize_files(
                intensity_files=[str(i_file)],
                mask_files=[str(seg_file[0])],
                single_roi=False,
                output_type=file_ext,
                output_path=str(output_path),
            )
        else:
            feat = nyx.featurize_files(
                intensity_files=[str(i_file)],
                mask_files=[str(seg_file[0])],
                single_roi=False,
            )
            feat.to_csv(str(output_path), index=False)


def run_nyxus_whole_image_features(
    int_file: pathlib.Path,
    out_dir: pathlib.Path,
    features: list[str],
    file_extension: Extension,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Extract Nyxus features for full intensity images.

    Args:
        int_file : Path to intensity image.
        out_dir : Path to output directory.
        features : List of features to compute.
        file_extension: Output file format (Extension enum or string).
        kwargs: Additional parameters passed to Nyxus.set_params().
    """
    nyx = Nyxus(features)

    if kwargs is None:
        kwargs = {}

    nyx.set_params(**kwargs)

    file_ext, suffix = _resolve_file_ext(file_extension)

    logger.info("Running Nyxus whole-image feature extraction")

    with BioReader(int_file) as br:
        image = br.read().squeeze()

    mask = np.ones(image.shape[:2], dtype=np.uint8)
    out_name = int_file.name.replace("".join(int_file.suffixes), suffix)
    output_path = pathlib.Path(out_dir, out_name)
    if file_ext != "pandas":
        nyx.featurize(image, mask, output_type=file_ext, output_path=str(output_path))
    else:
        feats = nyx.featurize(image, mask)
        feats.to_csv(str(output_path), index=False)
