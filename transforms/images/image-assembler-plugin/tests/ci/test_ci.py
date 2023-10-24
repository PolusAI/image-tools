from pathlib import Path
from typing import Tuple
from tests.ci.fixtures_ci import nist_mist_dataset
from tests.fixtures import plugin_dirs, ground_truth_dir


def test_image_assembler(nist_mist_dataset: Tuple[Path, Path]) -> None:
    """
    The reference nist mist dataset is composed of stripped tiff and
    won't be processed by bfio, so we cannot run this test unless
    we convert each fov first to ome tiled tiff.
    """
    pass

    # print("running our test")
    # img_path, stitch_path = nist_mist_dataset

    # print(f"created fixture : {img_path} \n {stitch_path}")

    # for img in img_path.iterdir():
    #     print(img.absolute())

    # for vector in stitch_path.iterdir():
    #     print(vector.absolute())

    # use file pattern2 to read the stitching vector
    # and get max x, y for top left corner of tiles

    # open fov with bfio and extract width, height
