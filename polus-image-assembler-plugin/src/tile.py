from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging
import numpy as np
from pathlib import Path
from main import _parse_stitch

def make_tile(x,X_range,y,Y_range,stitchPath,imgPath):
    # NOTE: This function should be rewritten to minimize the memory footprint.
    #       While the output image can be any size, the input image size will be limited by system memory.

    # Parse the stitching vector
    outvals = _parse_stitch(stitchPath,imgPath,True)

    # Get the data type
    br = BioReader(str(Path(imgPath).joinpath(outvals['filePos'][0]['file'])))
    dtype = br._pix['type']

    # initialize the supertile
    template = np.zeros((Y_range-y,X_range-x,1,1,1),dtype=dtype)

    # get images in bounds of current super tile
    for f in outvals['filePos']:
        if (f['posX'] > x and f['posX'] < X_range) or (f['posX']+f['width'] > x and f['posX']+f['width'] < X_range):
            if (f['posY'] > y and f['posY'] < Y_range) or (f['posY']+f['height'] > y and f['posY']+f['height'] < Y_range):
                br = BioReader(str(Path(imgPath).joinpath(f['file'])))
                image = br.read_image()
            else:
                continue
        else:
            continue
        
        # get bounds of image within the tile
        x_min = max(0,f['posX']-x)
        x_max = min(X_range-x,f['posX']+f['width']-x)
        y_min = max(0,f['posY']-y)
        y_max = min(Y_range-y,f['posY']+f['height']-y)

        # get bounds of image within the image
        xi = max(0,x - f['posX'])
        xe = min(f['width'],X_range - f['posX'])
        yi = max(0,y - f['posY'])
        ye = min(f['height'],Y_range - f['posY'])

        # write the image to the super tile
        template[y_min:y_max,x_min:x_max] = image[yi:ye,xi:xe]

    return template

if __name__=="__main__":

    # Setup the argument parsing
    parser = argparse.ArgumentParser(prog='assemble', description='Assemble images from a single stitching vector.')
    parser.add_argument('--stitchPath', dest='stitchPath', type=str,
                        help='Complete path to a stitching vector', required=True)
    parser.add_argument('--imgPath', dest='imgPath', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--x', dest='x', type=str,
                        help='Starting x value', required=True)
    parser.add_argument('--X_range', dest='X_range', type=str,
                        help='Maximum x value', required=True)
    parser.add_argument('--y', dest='y', type=str,
                        help='Starting y value', required=True)
    parser.add_argument('--Y_range', dest='Y_range', type=str,
                        help='Maximum y value', required=True)

    # Parse the arguments
    args = parser.parse_args()
    stitchPath = args.stitchPath
    imgPath = args.imgPath
    outDir = args.outDir
    x = int(args.x)
    X_range = int(args.X_range)
    y = int(args.y)
    Y_range = int(args.Y_range)

    # Start the javabridge with proper java logging
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    template = make_tile(x,X_range,y,Y_range,stitchPath,imgPath)

    bw = BioWriter(str(Path(outDir).joinpath('x{}_y{}.ome.tif'.format(x,y)).absolute()),image=template)
    bw.num_x(template.shape[1])
    bw.num_y(template.shape[0])
    bw.write_image(template)    
    bw.close_image()

    jutil.kill_vm()
    