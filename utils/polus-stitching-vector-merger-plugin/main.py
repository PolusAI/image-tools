import argparse, logging, time
from pathlib import Path
import shutil
import os


if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='This plugin merges stitching vector collections')
    parser.add_argument('--VectorCollection1', dest='VectorCollection1', type=str,
                        help='1st stitchingVector Collection', required=True)
    parser.add_argument('--VectorCollection2', dest='VectorCollection2', type=str,
                        help='2nd stitchingVector Collection', required=True)
    parser.add_argument('--VectorCollection3', dest='VectorCollection3', type=str,
                        help='3rd stitchingVector Collection', required=False)
    parser.add_argument('--VectorCollection4', dest='VectorCollection4', type=str,
                        help='4th stitchingVector Collection', required=False)
    parser.add_argument('--VectorCollection5', dest='VectorCollection5', type=str,
                        help='5th stitchingVector Collection', required=False)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    VectorCollection1 = args.VectorCollection1
    logger.info('VectorCollection1 = {}'.format(VectorCollection1))
    VectorCollection2 = args.VectorCollection2
    logger.info('VectorCollection2 = {}'.format(VectorCollection2))
    VectorCollection3 = args.VectorCollection3
    logger.info('VectorCollection3 = {}'.format(VectorCollection3))
    VectorCollection4 = args.VectorCollection4
    logger.info('VectorCollection4 = {}'.format(VectorCollection4))
    VectorCollection5 = args.VectorCollection5
    logger.info('VectorCollection5 = {}'.format(VectorCollection5))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    
    Collection_directories=[VectorCollection1,VectorCollection2,VectorCollection3,VectorCollection4,VectorCollection5]    
    
    count=0
    for inpDir in Collection_directories:
        if inpDir != None:
            for vector_name in sorted(os.listdir(inpDir)):            
                count+=1
                logger.info('Copying stitching vector : {} from {}'.format(vector_name,inpDir))
                shutil.copyfile(os.path.join(inpDir,vector_name),os.path.join(outDir,"img-global-positions-{}.txt".format(count)))
                

                
                
            
        
    
  
    
    
    
    
    
    
    
    
    
    
    

    
    
   
