from bfio.bfio import BioReader, BioWriter
import argparse, logging, typing
import numpy as np
from pathlib import Path
from filepattern import get_regex,FilePattern,VARIABLES
from preadator import ProcessManager

def unshade_image(img,out_dir,brightfield,darkfield,photobleach=None,offset=None):
    
    with ProcessManager.thread():
    
        with BioReader(img,max_workers=1) as br:
            
            with BioWriter(out_dir.joinpath(img.name),metadata=br.metadata,max_workers=2) as bw:
        
                new_img = br[:,:,:1,0,0].squeeze().astype(np.float32)
                
                new_img = new_img - darkfield
                new_img = np.divide(new_img,brightfield)
                
                if photobleach != None:
                    new_img = new_img - np.float32(photobleach)
                if offset != None:
                    new_img = new_img + np.float32(offset)

                new_img[new_img<0] = 0
                
                new_img = new_img.astype(br.dtype)
                
                bw[:] = new_img

def unshade_batch(files: typing.List[Path],
                  out_dir: Path,
                  brightfield: Path,
                  darkfield: Path,
                  photobleach: int = None,
                  offset: int = None):
    
    with ProcessManager.process():
    
        with BioReader(brightfield,max_workers=2) as bf:
            brightfield_image = bf[:,:,:,0,0].squeeze()
            
        with BioReader(darkfield,max_workers=2) as df:
            darkfield_image = df[:,:,:,0,0].squeeze()
        
        threads = []

        for file in files:
            
            ProcessManager.submit_thread(unshade_image,file['file'],
                                                       out_dir,
                                                       brightfield_image,
                                                       darkfield_image)
        
        ProcessManager.join_threads()

# Variables that will be grouped for the purposes of applying a flatfield
GROUPED = [v for v in 'xyp']
GROUPED.append('file')

if __name__=="__main__":
    ''' Initialize the logger '''
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    # Initialize the argument parser
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Apply a flatfield algorithm to a collection of images.')
    parser.add_argument('--darkPattern', dest='darkPattern', type=str,
                        help='Filename pattern used to match darkfield files to image files', required=False)
    parser.add_argument('--ffDir', dest='ffDir', type=str,
                        help='Image collection containing brightfield and/or darkfield images', required=True)
    parser.add_argument('--brightPattern', dest='brightPattern', type=str,
                        help='Filename pattern used to match brightfield files to image files', required=True)
    parser.add_argument('--imgDir', dest='imgDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--imgPattern', dest='imgPattern', type=str,
                        help='Filename pattern used to separate data and match with flatfied files', required=True)
    parser.add_argument('--photoPattern', dest='photoPattern', type=str,
                        help='Filename pattern used to match photobleach files to image files', required=False)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    darkPattern = args.darkPattern
    logger.info('darkPattern = {}'.format(darkPattern))
    ffDir = Path(args.ffDir)
    # catch the case that ffDir is the output within a workflow
    if Path(ffDir).joinpath('images').is_dir():
        ffDir = ffDir.joinpath('images')
    logger.info('ffDir = {}'.format(ffDir))
    brightPattern = args.brightPattern
    logger.info('brightPattern = {}'.format(brightPattern))
    imgDir = Path(args.imgDir)
    logger.info('imgDir = {}'.format(imgDir))
    imgPattern = args.imgPattern
    logger.info('imgPattern = {}'.format(imgPattern))
    photoPattern = args.photoPattern
    logger.info('photoPattern = {}'.format(photoPattern))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))

    ''' Start a process for each set of brightfield/darkfield/photobleach patterns '''
    # Create the FilePattern objects to handle file access
    ff_files = FilePattern(ffDir,brightPattern)
    fp = FilePattern(imgDir,imgPattern)
    if darkPattern != None and darkPattern!='':
        dark_files = FilePattern(ffDir,darkPattern)
    if photoPattern != None and photoPattern!='':
        photo_files = FilePattern(str(Path(ffDir).parents[0].joinpath('metadata').absolute()),photoPattern)

    ProcessManager.init_processes('main','unshade')
        
    for files in fp(group_by='xyp'):
        
        flat_path = ff_files.get_matching(**{k.upper():v for k,v in files[0].items() if k not in GROUPED})[0]['file']
        if flat_path == None:
            continue
        
        if darkPattern != None and darkPattern!='':
            dark_path = dark_files.get_matching(**{k.upper():v for k,v in files[0].items() if k not in GROUPED})[0]['file']
            
            if dark_path == None:
                continue
            
        ProcessManager.submit_process(unshade_batch,files,outDir,flat_path,dark_path)
    
    ProcessManager.join_processes()