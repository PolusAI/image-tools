from bfio import  BioWriter , OmeXml,JARS,LOG4J
#from bfio import  BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, sys
import numpy as np
from pathlib import Path
import re
import zarr
import mask

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')
    
    # Input arguments

    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--flow_threshold', required=False,
                        default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--cellprob_threshold', required=False,
                        default=0.0, type=float, help='cell probability threshold, centered at 0.0')
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()

    inpDir = args.inpDir


    logger.info('inpDir = {}'.format(inpDir))

    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    cellprob_threshold = args.cellprob_threshold
    flow_threshold= args.flow_threshold

    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing ...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
   #     jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=JARS)
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(LOG4J)],class_path=JARS)
        # Get all file names in inpDir image collection
        root = zarr.open(str(Path(inpDir).joinpath('location.zarr')),mode='r')
        i=1
        # Loop through files in inpDir image collection and process
        for m,l in root.groups():
            prob=l['pixel_location']
            loc = l['probablity']
            prob= np.asarray(prob)
            loc= np.asarray(loc)
            metadata=l.attrs['metadata']

            if len(prob.shape)==4:
                mask_stack=[]
                for i in range(int(prob.shape[0])):
                    masks = mask.compute_masks(y=prob[i,:,:,:],cellprob=loc[i,:,:,:],flow_threshold=flow_threshold,cellprob_threshold=cellprob_threshold)
                    mask_stack.append(masks)
                maski = np.asarray(mask_stack)
                maski = np.transpose(maski,(1,2,0))
                x,y,z = maski.shape
                maski = np.reshape(maski,(x,y,z,1,1))

            elif len(prob.shape) == 3 :

                maski = mask.compute_masks(y=prob[:],cellprob=loc[:])

                x,y = maski.shape
                maski = np.reshape(maski, (x,y, 1, 1,1))

            logger.info('Processing image ({}/{}): {}'.format(i,len([m for m,l in root.groups()]),m))

            # Write the output
            temp = re.sub(r'(?<=Type=")([^">>]+)', str(maski.dtype), metadata)
            xml_metadata = OmeXml.OMEXML(temp)
            path=Path(outDir).joinpath(str(str(m).split('.',1)[0]+'_mask.'+str(m).split('.',1)[1]))
            bw = BioWriter(file_path=Path(path),backend='python',max_workers=1,metadata=xml_metadata)
            bw.write(maski)
            bw.close()
            del maski
            i+=1
    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing ')
     #   jutil.kill_vm()
        
        # Exit the program
        sys.exit()