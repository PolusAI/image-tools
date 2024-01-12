"""Test Outlier Removal Plugin."""
import shutil
from pathlib import Path

import filepattern as fp
import numpy as np
import polus.plugins.clustering.outlier_removal.outlier_removal as rm
import vaex
from polus.plugins.clustering.outlier_removal.outlier_removal import Methods


def test_outlier_detection(
    generate_synthetic_data: tuple[Path, Path, str, Methods],
) -> None:
    """Test outlier detection of tabular data."""
    inp_dir, out_dir, file_extension, method = generate_synthetic_data

    method_type = "Global"
    file_pattern = f".*{file_extension}"
    files = fp.FilePattern(inp_dir, file_pattern)
    for file in files():
        rm.outlier_detection(
            file=file[1][0],
            method=f"{method}",
            out_dir=out_dir,
            method_type=method_type,
        )
    out_ext = [Path(f.name).suffix for f in out_dir.iterdir()]
    assert all(out_ext) is True
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


def test_isolationforest(
    generate_synthetic_data: tuple[Path, Path, str, Methods],
) -> None:
    """Test isolationforest method."""
    inp_dir, out_dir, file_extension, method = generate_synthetic_data
    file_pattern = f".*{file_extension}"
    files = fp.FilePattern(inp_dir, file_pattern)
    for file in files():
        df = vaex.open(file[1][0])
        data = df[df.column_names[:-1]].values
        prediction = rm.isolationforest(data, method)
        assert len(prediction) != 0
        assert type(prediction) == np.ndarray
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
