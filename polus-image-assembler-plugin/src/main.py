import argparse, logging, multiprocessing, re
from bfio import BioReader,BioWriter
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
STITCH_LINE = "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n"

def buffer_image(image_path,supertile_buffer,Xi,Yi,Xt,Yt):
    """buffer_image Load and image and store in buffer

    This method loads an image and stores it in the appropriate
    position based on the stitching vector coordinates within
    a large tile of the output image. It is intended to be
    used as a thread to increase the reading component to
    assembling the image.
    
    Args:
        image_path ([str]): Path to image to load
        supertile_buffer ([np.ndarray]): A supertile storing multiple images
        Xi ([list]): Xmin and Xmax of pixels to load from the image
        Yi ([list]): Ymin and Ymax of pixels to load from the image
        Xt ([list]): X position within the buffer to store the image
        Yt ([list]): Y position within the buffer to store the image
    """
    
    # Load the image
    br = BioReader(image_path,max_workers=2)
    image = br.read_image(X=Xi,Y=Yi) # only get the first z,c,t layer
        
    # Put the image in the buffer
    supertile_buffer[Yt[0]:Yt[1],Xt[0]:Xt[1],...] = image

def make_tile(x_min,x_max,y_min,y_max,stitchPath):
    """make_tile Create a supertile

    This method identifies images that have stitching vector positions
    within the bounds of the supertile defined by the x and y input
    arguments. It then spawns threads to load images and store in the
    supertile buffer. Finally it returns the assembled supertile to
    allow the main thread to generate the write thread.

    Args:
        x_min ([int]): Minimum x bound of the tile
        x_max ([int]): Maximum x bound of the tile
        y_min ([int]): Minimum y bound of the tile
        y_max ([int]): Maximum y bound of the tile
        stitchPath ([str]): Path to the stitching vector

    Returns:
        [type]: [description]
    """
    # Parse the stitching vector
    outvals = _parse_stitch(stitchPath,imgPath,True)

    # Get the data type
    br = BioReader(str(Path(imgPath).joinpath(outvals['filePos'][0]['file'])))
    dtype = br._pix['type']

    # initialize the supertile
    template = np.zeros((y_max-y_min,x_max-x_min,1,1,1),dtype=dtype)

    # get images in bounds of current super tile
    with ThreadPoolExecutor(max([multiprocessing.cpu_count(),2])) as executor:
        for f in outvals['filePos']:
            if (f['posX'] >= x_min and f['posX'] <= x_max) or (f['posX']+f['width'] >= x_min and f['posX']+f['width'] <= x_max):
                if (f['posY'] >= y_min and f['posY'] <= y_max) or (f['posY']+f['height'] >= y_min and f['posY']+f['height'] <= y_max):
            
                    # get bounds of image within the tile
                    Xt = [max(0,f['posX']-x_min)]
                    Xt.append(min(x_max-x_min,f['posX']+f['width']-x_min))
                    Yt = [max(0,f['posY']-y_min)]
                    Yt.append(min(y_max-y_min,f['posY']+f['height']-y_min))

                    # get bounds of image within the image
                    Xi = [max(0,x_min - f['posX'])]
                    Xi.append(min(f['width'],x_max - f['posX']))
                    Yi = [max(0,y_min - f['posY'])]
                    Yi.append(min(f['height'],y_max - f['posY']))
                    
                    executor.submit(buffer_image,str(Path(imgPath).joinpath(f['file'])),template,Xi,Yi,Xt,Yt)
    
    return template

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
        return int(s)
    except ValueError:
        return s

def _parse_stitch(stitchPath,imagePath,timepointName=False):
    """ Load and parse image stitching vectors
    
    This function creates a list of file dictionaries that include the filename and
    pixel position and dimensions within a stitched image. It also determines the
    size of the final stitched image and the suggested name of the output image based
    on differences in file names in the stitching vector.

    Inputs:
        stitchPath - A path to stitching vectors
        imagePath - A path to tiled tiff images
        timepointName - Use the vector timeslice as the image name
    Outputs:
        out_dict - Dictionary with keys (width, height, name, filePos)
    """

    # Initialize the output
    out_dict = { 'width': int(0),
                 'height': int(0),
                 'name': '',
                 'filePos': []}

    # Set the regular expression used to parse each line of the stitching vector
    line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"

    # Get a list of all images in imagePath
    images = [p.name for p in Path(imagePath).iterdir()]

    # Open each stitching vector
    fpath = str(Path(stitchPath).absolute())
    name_pos = {}
    with open(fpath,'r') as fr:

        # Read the first line to get the filename for comparison to all other filenames
        line = fr.readline()
        stitch_groups = re.match(line_regex,line)
        stitch_groups = {key:val for key,val in zip(STITCH_VARS,stitch_groups.groups())}
        name = stitch_groups['file']
        name_ind = [i for i in range(len(name))]
        fr.seek(0) # reset to the first line

        # Read each line in the stitching vector
        for line in fr:
            # Read and parse values from the current line
            stitch_groups = re.match(line_regex,line)
            stitch_groups = {key:get_number(val) for key,val in zip(STITCH_VARS,stitch_groups.groups())}
            
            # If an image in the vector doesn't match an image in the collection, then skip it
            if stitch_groups['file'] not in images:
                continue

            # Get the image size
            stitch_groups['width'], stitch_groups['height'] = BioReader.image_size(str(Path(imagePath).joinpath(stitch_groups['file']).absolute()))
            if out_dict['width'] < stitch_groups['width']+stitch_groups['posX']:
                out_dict['width'] = stitch_groups['width']+stitch_groups['posX']
            if out_dict['height'] < stitch_groups['height']+stitch_groups['posY']:
                out_dict['height'] = stitch_groups['height']+stitch_groups['posY']

            # Set the stitching vector values in the file dictionary
            out_dict['filePos'].append(stitch_groups)

            # Determine the difference between first name and current name
            if not timepointName:
                for i in name_ind:
                    if name[i] != stitch_groups['file'][i]:
                        if i not in name_pos.keys():
                            name_pos[i] = set()
                            name_pos[i].update([get_number(stitch_groups['file'][i])])
                            name_pos[i].update([get_number(name[i])])
                        else:
                            name_pos[i].update([get_number(stitch_groups['file'][i])])
    
    # Generate the output file name
    # NOTE: This should be rewritten later to determine numeric values rather than position values.
    #       Output file names should be 
    indices = sorted(name_pos.keys())
    if timepointName:
        global_regex = ".*global-positions-([0-9]+).txt"
        name = re.match(global_regex,Path(stitchPath).name).groups()[0]
        name += '.ome.tif'
        out_dict['name'] = name
    elif len(indices) > 0:
        out_dict['name'] = name[0:indices[0]]
        minvals = []
        maxvals = []
        for v,i in enumerate(indices):
            if len(minvals)==0:
                out_dict['name'] += '<'
            minvals.append(min(name_pos[i]))
            maxvals.append(max(name_pos[i]))
            if i == indices[-1] or indices[v+1] - i > 1:
                out_dict['name'] += ''.join([str(ind) for ind in minvals])
                out_dict['name'] += '-'
                out_dict['name'] += ''.join([str(ind) for ind in maxvals])
                out_dict['name'] += '>'
                if i ==  indices[-1]:
                    out_dict['name'] += name[indices[-1]+1:]
                else:
                    out_dict['name'] += name[indices[v]+1:indices[v+1]]
                minvals = []
                maxvals = []
    else:
        out_dict['name'] = name

    return out_dict

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Assemble images from a single stitching vector.')
    parser.add_argument('--stitchPath', dest='stitchPath', type=str,
                        help='Complete path to a stitching vector', required=True)
    parser.add_argument('--imgPath', dest='imgPath', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--timesliceNaming', dest='timesliceNaming', type=str,
                        help='Use timeslice number as image name', required=False)

    # Parse the arguments
    args = parser.parse_args()
    imgPath = args.imgPath
    if Path(imgPath).joinpath('images').is_dir():
        imgPath = str(Path(imgPath).joinpath('images').absolute())
    outDir = args.outDir
    logger.info('outDir: {}'.format(outDir))
    timesliceNaming = args.timesliceNaming == 'true'
    logger.info('timesliceNaming: {}'.format(timesliceNaming))
    stitchPath = args.stitchPath

    # Get a list of stitching vectors
    vectors = [str(p.absolute()) for p in Path(stitchPath).iterdir() if p.is_file() and "".join(p.suffixes)=='.txt']
        
    logger.info('imgPath: {}'.format(imgPath))
    logger.info('stitchPath: {}'.format(stitchPath))
    vectors.sort()

    # Variables for image building processes
    img_processes = []
    img_paths = []

    for v in vectors:
        # Check to see if the file is a stitching vector
        if 'img-global-positions' not in Path(v).name:
            continue
        
        # Parse the stitching vector
        logger.info('Analyzing vector: {}'.format(Path(v).name))
        outvals = _parse_stitch(v,imgPath,timesliceNaming)
        logger.info('Building image: {}'.format(outvals['name']))
        logger.info('Output image size (width, height): {},{}'.format(outvals['width'],outvals['height']))

        # Variables for tile building processes
        pnum = 0
        ptotal = np.ceil(outvals['width']/10240) * np.ceil(outvals['height']/10240)
        ptotal = 1/ptotal * 100
        
        # Initialize the output image
        logger.info('Initializing output file: {}'.format(outvals['name']))
        refImg = str(Path(imgPath).joinpath(outvals['filePos'][0]['file']).absolute())
        outFile = str(Path(outDir).joinpath(outvals['name']).absolute())
        br = BioReader(str(Path(refImg).absolute()))
        bw = BioWriter(str(Path(outFile).absolute()),metadata=br.read_metadata(),max_workers=max([multiprocessing.cpu_count(),2]))
        bw.num_x(outvals['width'])
        bw.num_y(outvals['height'])
        del br

        # Assemble the images
        logger.info('Generating tiles...')
        threads = []
        with ThreadPoolExecutor(max([multiprocessing.cpu_count()//2,2])) as executor:
            for x in range(0, outvals['width'], 10240):
                X_range = min(x+10240,outvals['width']) # max x-pixel index in the assembled image
                for y in range(0, outvals['height'], 10240):
                    Y_range = min(y+10240,outvals['height']) # max y-pixel index in the assembled image
                    
                    image_buffer = make_tile(x,X_range,y,Y_range,v)
                    
                    threads.append(executor.submit(bw.write_image,image_buffer,X=[x],Y=[y]))
                    # bw.write_image(image_buffer,X=[x],Y=[y])
            
            logger.info('{:.2f} finished...'.format(0))
            for ind,thread in enumerate(threads):
                thread.result()
                logger.info('{:.2f}% finished...'.format(100*(ind+1)/len(threads)))
        
        logger.info('Closing image...')
        bw.close_image()
