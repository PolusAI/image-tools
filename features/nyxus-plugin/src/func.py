import os
import pathlib 
import re
from typing import Optional, List
from nyxus import Nyxus
import logging
import itertools

from torch import is_deterministic_algorithms_warn_only_enabled

logger = logging.getLogger("main")
def nyxus_func(inpDir:str,
          segDir:str,
          outDir:pathlib.Path,
          filePattern:str,
          mapVar:List[str],
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
        mapVar list[str] : Variable containing image channel information in intensity images for feature extraction
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

    logger.info(f'filepattern is {filePattern}')

    nyx = Nyxus(features, 
                neighbor_distance=neighborDist, 
                n_feature_calc_threads=4,
                pixels_per_micron=pixelPerMicron
                )

    if mapVar:
        if len(mapVar) >1:
            mapVar = [f'{x}' for x in mapVar.split(',') if x in mapVar]
        else:
            mapVar = [mapVar]
        
        var = mapVar[0]
        var = re.match(r'\w', var).group(0)
        seglist = [re.findall(filePattern, f) for f in sorted(os.listdir(segDir))]

        # Filepattern is modified to extract features from other intensity channel images
        filePattern = re.sub(rf"{var}.*$", '.ome.tif', filePattern)
        intlist = [re.findall(filePattern, f) for f in sorted(os.listdir(inpDir))]
       
        intlist= list(itertools.chain(*intlist))
        seglist = list(itertools.chain(*seglist)) 

        logger.info(f'{intlist}')
        flist = []
        for m in mapVar:
            intval=[os.path.join(inpDir, v) for v in intlist if m in v if v.endswith('.ome.tif')]
            flist.append(intval)
        flist = sorted(list(itertools.chain(*flist)))
        segval = sorted([os.path.join(segDir, v) for v in sorted(seglist)] * len(mapVar))

        assert len(flist) == len(segval), logger.info('Unequal length of intensity filenames {flist} & label image {segval}!')

        features = nyx.featurize(int_fnames=flist,
                                seg_fnames=segval                               )
    else:
        features = nyx.featurize_directory(intensity_dir=inpDir,
                                label_dir=segDir,
                                file_pattern=filePattern
                                )
    outname = re.search(r'^\w', filePattern).group()
    outname = f'{outname}{replicate}.csv'
    features.to_csv(pathlib.Path(outDir, outname), index = False)

    return 