"""Test Hdbscan Clustering Plugin."""

import shutil
from pathlib import Path

import filepattern as fp
import polus.images.clustering.hdbscan_clustering.hdbscan_clustering as hd
import vaex


def test_hdbscan_clustering(generate_synthetic_data: tuple[Path, Path, str]) -> None:
    """Test hdbscan clustering of tabular data."""
    inp_dir, out_dir, file_extension = generate_synthetic_data
    pattern = r"\w+$"
    file_pattern = f".*{file_extension}"
    files = fp.FilePattern(inp_dir, file_pattern)
    for file in files():
        hd.hdbscan_clustering(
            file=file[1][0],
            min_cluster_size=3,
            grouping_pattern=pattern,
            label_col="species",
            average_groups=True,
            increment_outlier_id=True,
            out_dir=out_dir,
        )

    out_ext = [Path(f.name).suffix for f in out_dir.iterdir()]
    assert all(out_ext) is True
    for f in out_dir.iterdir():
        df = vaex.open(f)
        assert "cluster" in df.column_names
        assert df["cluster"].values != 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


def test_hdbscan_model(generate_synthetic_data: tuple[Path, Path, str]) -> None:
    """Test hdbscan model."""
    inp_dir, _, file_extension = generate_synthetic_data
    file_pattern = f".*{file_extension}"
    files = fp.FilePattern(inp_dir, file_pattern)
    for file in files():
        df = vaex.open(file[1][0])
        data = df[df.column_names[:-1]].values
        min_cluster_size = 3
        label = hd.hdbscan_model(data, min_cluster_size, True)
        assert len(label) != 0
    shutil.rmtree(inp_dir)
