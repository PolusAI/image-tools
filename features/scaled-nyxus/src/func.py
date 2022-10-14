import pathlib 
import re
from typing import Optional, List
from nyxus import Nyxus


def nyxus_func(inpDir:str,
          segDir:str,
          outDir:pathlib.Path,
          filePattern:str,
          features:List[str],
          neighborDist:Optional[float],
          pixelPerMicron:Optional[float],
          replicate:str
          ):

    """Scalable Extraction of Nyxus Features 
    Args:
        inpDir (str) : Path to intensity image directory
        segDir (str) : Path to label image directory
        outDir (Path) : Path to output directory
        features list[str] : List of nyxus features to be computed
        filePattern (str): Pattern to parse image replicates
        neighborDist (float) optional, default 5.0: Pixel distance between neighbor objects
        pixelPerMicron (float) optional, default 1.0: Pixel size in micrometer
        replicate (str) : replicate identified using filePattern and are processed in parallel
    Returns:
        df : pd.DataFrame
            Pandas DataFrame for each replicate containing extracted features for each object label

    """ 

    filePattern = (re.sub(r'\(.*?\)',
     '{replicate}', filePattern)
                    .format(replicate=replicate)
                    )

    nyx = Nyxus(features, 
                neighbor_distance=neighborDist, 
                n_feature_calc_threads=4,
                pixels_per_micron=pixelPerMicron
                )

    features = nyx.featurize(intensity_dir=inpDir,
                            label_dir=segDir,
                            file_pattern=filePattern
                            )

    outname = re.search(r'^\w', filePattern).group()
    outname = f'{outname}{replicate}.csv'
    features.to_csv(pathlib.Path(outDir, outname), index = False)

    return 