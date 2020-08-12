from bfio import BioWriter, BioReader
import bioformats
import javabridge as jutil
import argparse, logging, time, os, re, math
from filepattern import FilePattern
import numpy as np
from pathlib import Path

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
STITCH_LINE = "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n"

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def mean(data_list):
    """ Fast mean
    
    Calculates the artithmetic mean. This is faster than the method of the same name
    in the statistics package (1.5-10x speed improvement based on list size).

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The arithmetic mean of data_list
    """
    
    return sum(data_list)/len(data_list)

def count(data_list):
    """ Count number of objects

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The number of objects in an image
    """
    
    return len(data_list)

def var(data_list):
    """ Variance of the data

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The variance of the data
    """
    mean_sqr = mean(data_list)**2
    sqr_mean = mean([d**2 for d in data_list])
    return sqr_mean - mean_sqr

def median(data_list):
    """ Median of the data

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The median of the data
    """
    
    num_obj = count(data_list)
    data_list.sort()
    val = (data_list[(num_obj-1)//2] + data_list[num_obj//2]) / 2
    return val
    
def std(data_list):
    """ Standard deviation of the data

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The standard deviation of the data
    """
    
    return math.sqrt(var(data_list))
    
def skewness(data_list):
    """ Skewness of the data

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The skewness of the data
    """
    
    sigma = std(data_list)
    mu = mean(data_list)
    n = count(data_list)
    
    if n == 0 or sigma == 0:
        return 'NaN'
    
    skew = sum([(x-mu)**3 for x in data_list])/(n*sigma**3)
    
    return skew
    
def kurtosis(data_list):
    """ Kurtosis of the data

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The kurtosis of the data
    """
    
    sigma = std(data_list)
    mu = mean(data_list)
    n = count(data_list)
    
    if n == 0 or sigma == 0:
        return 'NaN'
    
    kurt = sum([(x-mu)**4 for x in data_list])/(n*sigma**4) - 3
    
    return kurt

def iqr(data_list):
    """ Interquartile range of the data

    Inputs:
        data_list - a list of floats
    Outputs:
        val - The interquartile range of the data
    """
    
    n = count(data_list)
    cnt = n//2
    
    if cnt ==0:
        return 'NaN'
    
    data_list.sort()
    l_half = data_list[:cnt]
    u_half = data_list[-cnt:]
    q1 = (l_half[(len(l_half)-1)//2] + l_half[len(l_half)//2]) / 2
    q3 = (u_half[(len(u_half)-1)//2] + u_half[len(u_half)//2]) / 2
    iqr = q3-q1
    return iqr

METHODS = {'mean': mean,
           'count': count,
           'var': var,
           'median': median,
           'std': std,
           'skewness': skewness,
           'kurtosis': kurtosis,
           'iqr': iqr}

def get_number(s):
    """ Check that s is number
    
    In this plugin, heatmaps are created only for columns that contain numbers. This
    function checks to make sure an input value is able to be converted into a number.

    Inputs:
        s - An input string or number
    Outputs:
        value - Either float(s) or False if s cannot be cast to float
    """
    try:
        s = float(s)
        if s==float('-inf') or s==float('inf') or str(s)=='nan':
            return False
        else:
            return float(s)
    except ValueError:
        return False

def _get_file_dict(fp,fname):
    """ Find an image matching fname in the collection
    
    This function searches files in a FilePattern object to find the image dictionary
    that matches the file name, fname.

    Inputs:
        fp - A FilePattern object
        fname - The name of the file to find in fp
    Outputs:
        current_image - The image dictionary matching fname, None if no matches found
    """
    current_image = None
    for f in fp.iterate():
        if Path(f['file']).name == fname:
            current_image = f
            break
    
    return current_image

def _parse_stitch(stitchPath,fp):
    """ Load and parse image stitching vectors
    
    This function adds keys to the FilePattern object (fp) that indicate image positions
    extracted from the stitching vectors found at the stitchPath location.

    As the stitching vector is parsed, images in the stitching vector are analyzed to
    determine the unique sets of image widths and heights. This information is required
    when generating the heatmap images to create overlays that are identical in size to
    the images in the original pyramid.

    Inputs:
        fp - A FilePattern object
        stitchPath - A path to stitching vectors
    Outputs:
        unique_width - List of all unique widths (in pixels) in the image stitching vectors
        unique_height - List of all unique heights (in pixels) in the image stitching vectors
    """
    # Get the stitch files
    txt_files = [f.name for f in Path(stitchPath).iterdir() if f.is_file() and f.suffix=='.txt']
    global_regex = ".*-global-positions-([0-9]+).txt"
    stitch_vectors = [re.match(global_regex,f) for f in txt_files if re.match(global_regex,f) is not None]
    
    line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"

    unique_width = set()
    unique_height = set()

    # Open each stitching vector
    fnum = 0
    fcheck = 0
    for vector in stitch_vectors:
        vind = vector.groups()[0]
        fpath = os.path.join(stitchPath,vector.group(0))
        with open(fpath,'r') as fr:

            # Read each line in the stitching vector
            line_num = 0
            for line in fr:
                # Read and parse values from the current line
                stitch_groups = re.match(line_regex,line)
                stitch_groups = {key:val for key,val in zip(STITCH_VARS,stitch_groups.groups())}

                # Get the image dictionary associated with the current line
                current_image = _get_file_dict(fp,stitch_groups['file'])
                
                # If an image in the vector doesn't match an image in the collection, then skip it
                if current_image == None:
                    continue

                # Set the stitching vector values in the file dictionary
                current_image.update({key:val for key,val in stitch_groups.items() if key != 'file'})
                current_image['vector'] = vind
                current_image['line'] = line_num
                line_num += 1

                # Get the image size
                current_image['width'], current_image['height'] = BioReader.image_size(current_image['file'])
                unique_height.update([current_image['height']])
                unique_width.update([current_image['width']])

                # Checkpoint
                fnum += 1
                if fnum//1000 > fcheck:
                    fcheck += 1
                    logger.info('Files parsed: {}'.format(fnum))

    return unique_width,unique_height

def _parse_features(featurePath,fp,method):
    """ Load and parse the feature list
    
    This function adds mean feature values to the FilePattern object (fp) for every image
    in the FilePattern object if the image is listed in the feature csv file.

    For example, if there are 100 object values in an "area" column for one image, then
    an "area" key is created in the image dictionary with the mean value of all 100 values.

    Inputs:
        fp - A FilePattern object
        stitchPath - A path to stitching vectors
    Outputs:
        unique_width - List of all unique widths (in pixels) in the image stitching vectors
        unique_height - List of all unique heights (in pixels) in the image stitching vectors
    """
    # Get the csv files containing features
    csv_files = [f.name for f in Path(featurePath).iterdir() if f.is_file() and f.suffix=='.csv']

    # Unique list of features and values
    feature_list = {}
    
    # Open each csv files
    for feat_file in csv_files:
        fpath = os.path.join(featurePath,feat_file)
        with open(fpath,'r') as fr:
            # Read the first line, which should contain headers
            first_line = fr.readline()
            headers = first_line.rstrip('\n').split(',')
            var_ind = {key:val for key,val in enumerate(headers)} # map headers to line positions

            # Add unique features to the feature_list
            feature_list.update({key:[] for key in headers if key not in feature_list.keys() and key != 'file'})

            # Get the first line of data
            line = fr.readline()

            # Read each line in the stitching vector
            fnum = 0
            fcheck = 0
            while line:
                # Parse the current line as a dictionary
                p_line = {var_ind[ind]:val for ind,val in enumerate(line.rstrip('\n').split(','))}
                for key,val in p_line.items():
                    v = get_number(val)
                    if isinstance(v,float):
                        p_line[key] = [v]
                    elif key != 'file':
                        p_line[key] = []

                # Get the image associated with the current line
                current_image = _get_file_dict(fp,p_line['file'])

                if current_image == None or 'line' not in current_image.keys():
                    line = fr.readline()
                    continue

                # Loop through rows until the filename changes
                line = fr.readline()
                np_line = {var_ind[ind]:val for ind,val in enumerate(line.rstrip('\n').split(','))}
                while line and p_line['file'] == np_line['file']:
                    # Store the values in a feature list
                    for key,val in np_line.items():
                        v = get_number(val)
                        if isinstance(v,float):
                            p_line[key].append(float(val))

                    # Get the next line
                    line = fr.readline()
                    np_line = {var_ind[ind]:val for ind,val in enumerate(line.rstrip('\n').split(','))}

                # Get the mean of the feature list, save in the file dictionary
                for key,val in p_line.items():
                    if isinstance(val,list):
                        try:
                            current_image[key] = METHODS[method](val)
                            feature_list[key].append(current_image[key])
                        except ZeroDivisionError:
                            current_image[key] = 'NaN'
                
                # Checkpoint
                fnum += 1
                if fnum//1000 > fcheck:
                    fcheck += 1
                    logger.info('Files parsed: {}'.format(fnum))

    return feature_list

if __name__=="__main__":
    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Build a heatmap pyramid for features values in a csv as an overlay for another pyramid.')
    parser.add_argument('--features', dest='features', type=str,
                        help='CSV collection containing features', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection used to build a pyramid that this plugin will make an overlay for', required=True)
    parser.add_argument('--method', dest='method', type=str,
                        help='Method used to create the heatmap', required=True)
    parser.add_argument('--vector', dest='vector', type=str,
                        help='Stitching vector used to buld the image pyramid.', required=True)
    parser.add_argument('--outImages', dest='outImages', type=str,
                        help='Heatmap Output Images', required=True)
    parser.add_argument('--vectorInMetadata', dest='vectorInMetadata', type=str,
                        help='Store stitching vector in metadata', required=True)
    parser.add_argument('--outVectors', dest='outVectors', type=str,
                        help='Heatmap Output Vectors', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    features = args.features
    logger.info('features = {}'.format(features))
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    method = args.method
    logger.info('method = {}'.format(method))
    vector = args.vector
    logger.info('vector = {}'.format(vector))
    outImages = args.outImages
    vectorInMetadata = args.vectorInMetadata == 'true'
    logger.info('vectorInMetadata = {}'.format(vectorInMetadata))
    if vectorInMetadata:
        outVectors = Path(outImages).joinpath('metadata_files')
        outVectors.mkdir()
        outVectors = str(outVectors.absolute())
        outImages = Path(outImages).joinpath('images')
        outImages.mkdir()
        outImages = str(outImages.absolute())
    else:
        outVectors = args.outVectors
    logger.info('outImages = {}'.format(outImages))
    logger.info('outVectors = {}'.format(outVectors))

    # Set up the fileparser
    fp = FilePattern(inpDir,'.*.ome.tif')

    # Parse the stitching vector
    logger.info('Parsing stitching vectors...')
    widths, heights = _parse_stitch(vector,fp)

    # Parse the features
    logger.info('Parsing features...')
    feature_list = _parse_features(features,fp,method)

    # Determine the min, max, and unique values for each data set
    logger.info('Setting feature scales...')
    feature_mins = {}
    feature_ranges = {}
    for key,val in feature_list.items():
        valid_vals = [v for v in val if v is not 'NaN']
        if len(valid_vals) == 0:
            feature_mins[key] = 0
            feature_ranges[key] = 0
        else:
            feature_mins[key] = min(valid_vals)
            feature_ranges[key] = max(valid_vals)-feature_mins[key]
    unique_levels = set()
    for fl in fp.iterate():
        if 'line' not in fl.keys():
            continue
        for ft in feature_list:
            try:
                if get_number(fl[ft]):
                    fl[ft] = round((fl[ft] - feature_mins[ft])/feature_ranges[ft] * 254 + 1)
                    unique_levels.update([fl[ft]])
                else:
                    fl[ft] = 0
                    unique_levels.update([0])
            except ZeroDivisionError:
                fl[ft] = 0
                unique_levels.update([0])
                
    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    # Generate the heatmap images
    logger.info('Generating heatmap images...')
    for w in widths:
        for h in heights:
            for l in unique_levels:
                out_file = Path(outImages).joinpath(str(w) + '_' + str(h) + '_' + str(l) + '.ome.tif')
                if not out_file.exists():
                    image = np.ones((h,w,1,1,1),dtype=np.uint8)*l
                    bw = BioWriter(str(Path(outImages).joinpath(str(w) + '_' + str(h) + '_' + str(l) + '.ome.tif').absolute()),X=w,Y=h,Z=1,C=1,T=1)
                    bw.write_image(image)
                    bw.close_image()

    # Close the javabridge
    logger.info('Closing the javabridge...')
    jutil.kill_vm()

    # Build the output stitching vector
    logger.info('Generating the heatmap...')
    file_name = '{}_{}_{}.ome.tif'
    for num,feat in enumerate(feature_list):
        fpath = str(Path(outVectors).joinpath('img-global-positions-' + str(num+1) + '.txt').absolute())
        with open(fpath,'w') as fw:
            line = 0
            while True:
                for f in fp.iterate():
                    if 'line' in f and f['line'] == line:
                        break
                if 'line' in f and f['line'] == line:
                    fw.write("file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n".format(file_name.format(f['width'],f['height'],f[feat]),
                                                                                                f['correlation'],
                                                                                                f['posX'],
                                                                                                f['posY'],
                                                                                                f['gridX'],
                                                                                                f['gridY']))
                    line += 1
                else:
                    break