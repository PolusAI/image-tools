"""Nyxus Plugin."""
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import vaex
from bfio import BioReader
from nyxus import Nyxus
from pyarrow import ipc

from .utils import Extension

logger = logging.getLogger(__name__)

chunk_size = 100_000

_EXT_MAP: dict[str, str] = {
    "arrowipc": ".arrow",
    "parquet": ".parquet",
    "pandas": ".csv",
}


def _resolve_file_ext(file_extension: Extension) -> tuple[str, str]:
    """Resolve a file extension enum/string to (file_ext, suffix).

    Args:
        file_extension: Extension enum value or plain string.

    Returns:
        A tuple of (file_ext, suffix), e.g. ("arrowipc", ".arrow").

    Raises:
        ValueError: If the extension is not supported.
    """
    if hasattr(file_extension, "value"):
        file_ext = str(file_extension.value).lower().strip()
    else:
        file_ext = str(file_extension).lower().strip()

    try:
        suffix = _EXT_MAP[file_ext]
    except KeyError as err:
        msg = f"Invalid extension '{file_extension}'. Options: {list(_EXT_MAP)}"
        raise ValueError(
            msg,
        ) from err

    return file_ext, suffix


def _write_features(
    feats: pd.DataFrame | str,
    file_ext: str,
    output_path: pathlib.Path,
) -> None:
    """Write a features DataFrame to disk in the requested format."""
    if isinstance(feats, str):
        feats_path = pathlib.Path(feats)

        if feats_path.suffix == ".csv":
            feats = pd.read_csv(feats_path)

        elif feats_path.suffix in [".arrow", ".feather"]:
            # Use normal open() instead of memory_map
            with feats_path.open("rb") as f:
                feats = ipc.open_file(f).read_all().to_pandas()

        elif feats_path.suffix == ".parquet":
            feats = pq.read_table(str(feats_path)).to_pandas()

        else:
            msg = f"Unsupported Nyxus output format: {feats_path}"
            raise ValueError(msg)

    # ---- Write output ----
    if file_ext == "pandas":
        vf = vaex.from_pandas(feats)
        vf.export_csv(path=str(output_path), chunk_size=100_000)

    else:
        table = pa.Table.from_pandas(feats)

        if file_ext == "arrowipc":
            with pa.OSFile(str(output_path), "wb") as sink, ipc.new_file(
                sink,
                table.schema,
            ) as writer:
                writer.write(table)

        elif file_ext == "parquet":
            pq.write_table(table, str(output_path))


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

        feats = nyx.featurize_files(
            intensity_files=[str(i_file)],
            mask_files=[str(seg_file[0])],
            single_roi=False,
            output_type=file_ext,
            output_path=str(output_path),
        )

        _write_features(feats, file_ext, output_path)


def run_nyxus_whole_image_features(
    int_file: pathlib.Path,
    out_dir: pathlib.Path,
    features: list[str],
    file_extension: Extension,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Extract Nyxus features for full intensity images."""
    nyx = Nyxus(features)

    if kwargs is None:
        kwargs = {}

    nyx.set_params(**kwargs)

    file_ext, suffix = _resolve_file_ext(file_extension)

    logger.info("Running Nyxus whole-image feature extraction")

    with BioReader(int_file) as br:
        image = br.read()

    mask = np.ones(image.shape[:2], dtype=np.uint8)

    feats = nyx.featurize(image, mask)

    out_name = int_file.name.replace("".join(int_file.suffixes), suffix)
    output_path = pathlib.Path(out_dir, out_name)

    _write_features(feats, file_ext, output_path)
