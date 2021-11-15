from bfio import BioReader
import argparse, logging
import numpy as np
from pathlib import Path
import filepattern, multiprocessing, utils
from concurrent.futures import ThreadPoolExecutor

COLORS = ['red',
          'green',
          'blue',
          'yellow',
          'magenta',
          'cyan',
          'gray']

def get_number(s):
    """ Check that s is number
    
    If s is a number, first attempt to convert it to an int.
    If integer conversion fails, attempt to convert it to a float.
    If float conversion fails, return None.
    
    Inputs:
        s - An input string or number
    Outputs:
        value - Either float, int or None
    """
    try:
        return [int(si) for si in s.split('-')]
    except ValueError:
        try:
            return [float(si) for si in s.split('-')]
        except ValueError:
            return None

def get_bounds(br,lower_bound,upper_bound):
    """ Calculate upper and lower pixel values for image rescaling
    
    This method calculates the upper and lower percentiles
    for a given image. The lower_bound and upper_bound must
    be floats in the range 0-1, where 0.01 indicates 1%. The
    values returned are pixel intensity values.
    
    Images are read in tiles to prevent images from being
    completely read into memory. This permits calculation
    of percentiles on images that are larger than memory.

    Args:
        br (bfio.BioReader): A BioReader object to access a tiled tiff
        lower_bound (float): Lower bound percentile, must be in 0.00-1.00
        upper_bound (float): Upper bound percentile, must be in 0.00-1.00

    Returns:
        [list]: List of upper and lower bound values in pixel intensity units.
    """
    # TODO: Replace pixel buffer with histogram/fingerprint to handle
    #       larger images and/or larger percentile values
    
    # Make sure the inputs are properly formatted
    assert isinstance(lower_bound,float) and isinstance(upper_bound,float)
    assert lower_bound >= 0 and lower_bound <= 1.0
    assert upper_bound >= 0 and upper_bound <= 1.0
    
    # Get the image size in pixels
    image_size = br.num_x() * br.num_y()
    
    # Get number of pixels needed to get percentile information
    upper_bound_size = int(image_size * (1-upper_bound))
    lower_bound_size = int(image_size * lower_bound)
    
    # Create the pixel buffer
    dtype = br.read_image(X=[0,1024],Y=[0,1024],Z=[0,1]).dtype
    upper_bound_vals = np.zeros((2*upper_bound_size,),dtype=dtype)
    lower_bound_vals = np.full((2*lower_bound_size,),np.iinfo(dtype).max,dtype=dtype)
    
    # Load image tiles and sort pixels
    for x in range(0,br.num_x(),8192):
        for y in range(0,br.num_y(),8192):
            
            # Load the first tile
            tile = br.read_image(X=[x,min([x+8192,br.num_x()])],
                                 Y=[y,min([y+8192,br.num_y()])],
                                 Z=[0,1])
            
            # Sort the non-zero values
            tile_sorted = np.sort(tile[tile.nonzero()],axis=None)

            # Store the upper and lower bound pixel values
            temp = tile_sorted[-upper_bound_size:]
            upper_bound_vals[:temp.size] = temp
            temp = tile_sorted[:lower_bound_size]
            lower_bound_vals[-temp.size:] = temp
            
            # Resort the pixels
            upper_bound_vals = np.sort(upper_bound_vals,axis=None)
            lower_bound_vals = np.sort(lower_bound_vals,axis=None)
    
    return [lower_bound_vals[lower_bound_size],upper_bound_vals[-upper_bound_size]]

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Builds a DeepZoom color pyramid.')
    
    # Input arguments
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--layout', dest='layout', type=str,
                        help='Color ordering (e.g. 1,11,,,,5,6)', required=True)
    parser.add_argument('--bounds', dest='bounds', type=str,
                        help='Set bounds (should be float-float, int-int, or blank, e.g. 0.01-0.99,0-16000,,,,,)', required=False)
    parser.add_argument('--alpha', dest='alpha', type=str,
                        help='If true, transparency is equal to pixel intensity in the pyramid.', required=False)
    parser.add_argument('--stitchPath', dest='stitch_path', type=str,
                        help='Path to a stitching vector.', required=False)
    parser.add_argument('--background', dest='background', type=str,
                        help='Background fill value.', required=False)
    
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output pyramid path.', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    layout = args.layout
    logger.info('layout = {}'.format(layout))
    bounds = args.bounds
    logger.info('bounds = {}'.format(bounds))
    alpha = args.alpha == 'true'
    logger.info('alpha = {}'.format(alpha))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    outDir = Path(outDir)
    stitch_path = args.stitch_path
    logger.info('stitchPath = {}'.format(stitch_path))
    background = args.background
    logger.info('background = {}'.format(background))
    
    # Parse the layout
    layout = [None if l=='' else int(l) for l in layout.split(',')]
    if len(layout)>7:
        layout = layout[:7]
        
    # Parse the bounds
    if bounds != None:
        bounds = [[None] if l=='' else get_number(l) for l in bounds.split(',')]
        bounds = bounds[:len(layout)]
    else:
        bounds = [[None] for _ in layout]
        
    # Parse files
    fp = filepattern.FilePattern(inpDir,filePattern)
    
    # A channel variable is expected, throw an error if it doesn't exist
    if 'c' not in fp.variables:
        raise ValueError('A channel variable is expected in the filepattern.')
    
    count = 0
        
    for files in fp.iterate(group_by=fp.variables):
        
        count += 1
        outDirFrame = outDir.joinpath('{}_files'.format(count))
        outDirFrame.mkdir()
        bioreaders = []
        threads = []
        with ThreadPoolExecutor(max([multiprocessing.cpu_count()//2,2])) as executor:
            for i,l in enumerate(layout):
                if l == None:
                    bioreaders.append(None)
                    continue
                
                # Create the type of BioReader based on whether a stitching vector is present
                if stitch_path == None:
                    # Create a standard BioReader object
                    f_path = [f for f in files if f['c']==l]
                    if len(f_path)==0:
                        bioreaders.append(None)
                        continue
                    f_path = f_path[0]['file']
                    bioreaders.append(BioReader(f_path,max_workers=multiprocessing.cpu_count()))
                else:
                    # Create a BioAssembler object, which assembles images when called
                    f_tiles = [f for f in files if f['c']==l]
                    if len(f_tiles)==0:
                        continue
                    for stitch in Path(stitch_path).iterdir():
                        br = utils.BioAssembler(inpDir,stitch,multiprocessing.cpu_count())
                        f_names = [f['file'] for f in br._file_dict['filePos']]
                        found_file = False
                        
                        # Check to see if the first file is in the stitching vector
                        if Path(f_tiles[0]['file']).name in f_names:
                            bioreaders.append(br)
                            found_file = True
                            break
                    if not found_file:
                        bioreaders.append(None)
                        continue
                    
                # Set the rescaling bounds
                if layout[i] != None:
                    if isinstance(bounds[i][0],float):
                        logger.info('{}: Getting percentile bounds {}...'.format(Path(bioreaders[-1]._file_path).name,
                                                                                bounds[i]))
                        # get_bounds(bioreaders[-1],bounds[i][0],bounds[i][1])
                        threads.append(executor.submit(get_bounds,bioreaders[-1],bounds[i][0],bounds[i][1]))
                    elif isinstance(bounds[i][0],int):
                        bioreaders[-1].bounds = bounds[i]
                    else:
                        bioreaders[-1].bounds = [np.iinfo(bioreaders[-1].read_metadata().image().Pixels.get_PixelType()).min,
                                                 np.iinfo(bioreaders[-1].read_metadata().image().Pixels.get_PixelType()).max]
            
            for i in reversed(range(len(layout))):
                if bioreaders[i] == None:
                    continue
                if isinstance(bounds[i][0],int) or bounds[i][0] == None:
                    logger.info('Color {}: {} (rescaling to {})'.format(COLORS[i],
                                                                        Path(Path(bioreaders[i]._file_path).name).name,
                                                                        bioreaders[i].bounds))
                    continue
                if layout[i] == None:
                    continue
                bioreaders[i].bounds = threads.pop().result()
                logger.info('Color {}: {} (rescaling to {})'.format(COLORS[i],
                                                                    Path(Path(bioreaders[i]._file_path).name).name,
                                                                    bioreaders[i].bounds))
                    
        for br in bioreaders:
            if br != None:
                br_meta = br
        file_info = utils.dzi_file(br_meta,outDirFrame,count)
        encoder = utils.DeepZoomChunkEncoder(file_info)
        file_writer = utils.DeepZoomWriter(outDirFrame)
        
        utils._get_higher_res(0,bioreaders,file_writer,encoder,alpha,background,isinstance(stitch_path,str))
