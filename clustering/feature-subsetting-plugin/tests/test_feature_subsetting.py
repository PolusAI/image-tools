"""Test Feature Subsetting Plugin."""
import shutil
from pathlib import Path

import polus.plugins.clustering.feature_subsetting.feature_subset as fs


def test_feature_subset(
    generate_synthetic_data: tuple[Path, Path, Path, str],
) -> None:
    """Test images subsetting based on feature values."""
    inp_dir, tabular_dir, out_dir, _ = generate_synthetic_data
    file_pattern = "x{x+}_y{y+}_p{p+}_c{c+}.ome.tif"
    image_feature = "intensity_image"
    tabular_feature = "MEAN"
    padding = 0
    percentile = 0.8
    remove_direction = "Below"
    group_var = "p,c"
    write_output = True

    fs.feature_subset(
        inp_dir=inp_dir,
        tabular_dir=tabular_dir,
        out_dir=out_dir,
        file_pattern=file_pattern,
        group_var=group_var,
        percentile=percentile,
        remove_direction=remove_direction,
        section_var=None,
        image_feature=image_feature,
        tabular_feature=tabular_feature,
        padding=padding,
        write_output=write_output,
    )

    out_ext = [Path(f.name).suffix for f in out_dir.iterdir()]
    assert len(out_ext) != 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
    shutil.rmtree(tabular_dir)


def test_filter_planes() -> None:
    """Test filter planes."""
    feature_dict = {
        1: 1236.597914951989,
        2: 1153.754875685871,
        3: 1537.3429175240055,
        4: 1626.0415809327849,
    }

    percentile = 0.1
    remove_direction = "Below"
    fn = fs.filter_planes(
        feature_dict=feature_dict,
        remove_direction=remove_direction,
        percentile=percentile,
    )

    assert type(fn) == set


def test_make_uniform() -> None:
    """Test each section contain same number of images."""
    planes_dict = {1: [3, 4]}
    uniques = [1, 2, 3, 4]
    padding = 0
    fn = fs.make_uniform(planes_dict=planes_dict, uniques=uniques, padding=padding)

    assert len(fn) != 0
