import bioformats,csv,logging,argparse
from bfio.bfio import BioReader,BioWriter
import javabridge as jutil
import numpy as np
from filepattern import FilePattern
from pathlib import Path

def _unshade(img,brightfield,darkfield,photobleach=None,offset=None):
    new_img = img.astype(np.float32)
    
    new_img = new_img - darkfield
    new_img = np.divide(new_img,brightfield)
    
    if photobleach != None:
        new_img = new_img - np.float32(photobleach)
    if offset != None:
        new_img = new_img + np.float32(offset)

    new_img[new_img<0] = 0

    return new_img.astype(img.dtype)

if __name__=="__main__":

    ''' Argument parsing '''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='main', description='Apply a flatfield algorithm to a collection of images.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Directory containing image to apply the flatfielding operation.', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Directory where the images should be saved.', required=True)
    parser.add_argument('--filepattern', dest='filepattern', type=str,
                        help='Filepattern used to parse the input file names.', required=True)
    parser.add_argument('--brightfield', dest='brightfield', type=str,
                        help='Path to the brightfield image.', required=True)
    parser.add_argument('--darkfield', dest='darkfield', type=str,
                        help='Path to the darkfield image.', required=False)
    parser.add_argument('--photobleach', dest='photobleach', type=str,
                        help='Path to the photobleach csv.', required=False)
    parser.add_argument('--R', dest='R', type=str,
                        help='Replicate to process.', required=True)
    parser.add_argument('--T', dest='T', type=str,
                        help='Timepoint to process.', required=True)
    parser.add_argument('--C', dest='C', type=str,
                        help='Channel to process.', required=True)

    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    outDir = args.outDir
    filepattern = args.filepattern
    brightfield = args.brightfield
    darkfield = args.darkfield
    photobleach = args.photobleach
    R = int(args.R)
    T = int(args.T)
    C = int(args.C)

    ''' Initialize the logger '''
    logging.basicConfig(format='%(asctime)s - %(name)-8s - Process [{0},{1},{2}] - %(levelname)-8s - %(message)s'.format(R,T,C),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("do_flat")
    logger.setLevel(logging.INFO)

    # Set up the FilePattern object
    images = FilePattern(inpDir,filepattern)

    ''' Start the javabridge '''
    logger.info("Starting the javabridge...")
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    ''' Load the flatfielding data '''
    logger.info("Loading the flatfield data...")
    flat_br = BioReader(brightfield)
    flat_image = np.squeeze(flat_br.read_image())
    del flat_br

    # Normalize the brightfield image if it isn't done already
    flat_image = flat_image.astype(np.float32)
    flat_image = np.divide(flat_image,np.mean(flat_image))

    # Load the darkfield and photobleach offsets if they are specified
    if darkfield != None:
        dark_br = BioReader(darkfield)
        dark_image = np.squeeze(dark_br.read_image())
        del dark_br
    else:
        dark_image = np.zeros(flat_image.shape,dtype=np.float32)
    if photobleach != None:
        with open(photobleach,'r') as f:
            reader = csv.reader(f)
            photo_offset = {line[0]:float(line[1]) for line in reader if line[0] != 'file'}
        offset = np.mean([o for o in photo_offset.values()])

    ''' Apply flatfield to images '''
    for f in images.iterate(R=R,C=C,T=T):
        p = Path(f[0]['file'])
        logger.info("Applying flatfield to image: {}".format(p.name))
        br = BioReader(str(p.absolute()))
        image = br.read_image()
        if photobleach != None:
            new_image = _unshade(np.squeeze(image),flat_image,dark_image,photo_offset[p.name],offset=offset)
        else:
            new_image = _unshade(np.squeeze(image),flat_image,dark_image)
        bw = BioWriter(str(Path(outDir).joinpath(p.name).absolute()),
                       metadata=br.read_metadata())
        bw.write_image(np.reshape(new_image,image.shape))
        bw.close_image()
        del br

    ''' Close the javabridge '''
    logger.info("Closing the javabridge and ending process...")
    jutil.kill_vm()