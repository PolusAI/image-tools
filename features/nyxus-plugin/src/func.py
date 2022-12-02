import pathlib
import os
from typing import List, Optional, Union
import logging
from nyxus import Nyxus
from preadator import ProcessManager

logger = logging.getLogger("func")


def nyxus_func(
    int_file: Union[pathlib.Path, List[pathlib.Path]],
    seg_file: pathlib.Path,
    out_dir: pathlib.Path,
    features: List[str],
    pixels_per_micron: float,
    neighbor_dist: Optional[float] = 5.0,
) -> None:
    """Scalable Extraction of Nyxus Features

    Args:
        int_file (Union[pathlib.Path, List[pathlib.Path]]): Path to intensity image(s)
        seg_file (pathlib.Path): Path to label image
        out_file (pathlib.Path): Path to output directory
        features (List[str]): Pattern to parse image replicates
        pixels_per_micron (float): Number of pixels for every micrometer
        neighbor_dist (Optional[float], optional): Pixel distance between neighbor
            objects. Defaults to 5.0.
    """

    with ProcessManager.process(seg_file.name):

        if isinstance(int_file, pathlib.Path):
            int_file = [int_file]

        nyx = Nyxus(
            features,
            neighbor_distance=neighbor_dist,
            n_feature_calc_threads=4,
            pixels_per_micron=pixels_per_micron,
        )

        for i_file in int_file:

            feats = nyx.featurize(
                intensity_files=[str(i_file.absolute())],
                mask_files=[str(seg_file.absolute())],
            )

            out_name = i_file.name.replace("".join(i_file.suffixes), ".csv")
            feats.to_csv(os.path.join(out_dir, out_name), index=False)
