from bfio import BioReader, BioWriter, JARS
import javabridge as jutil
import bioformats
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
import tifffile
import cv2
import ast
from scipy import ndimage
from multiprocessing import cpu_count

Tile_Size = 1024


if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    log_config = Path(__file__).parent.joinpath("log4j.properties")

    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    # Get all file names in inpDir image collection
    # inpDir_files = [f.name for f in Path(inpDir).iterdir()]
    # logger.info("Files in input directory: {}".format(inpDir_files))
    try:
        og = Path("/home/ec2-user/binaryimages/ogx003_y017_c001.ome.tif")
        file1 = Path("/home/ec2-user/polusBINARY/polus-plugins/polus-binary-operations-plugin/out/checkimage.ome.tif")
        file2 = Path("/home/ec2-user/polusBINARY/polus-plugins/polus-binary-operations-plugin/out/ogx003_y017_c001.ome.tif")

        logger.info(file1)
        logger.info(file2)

        # br0 = BioReader(og.absolute(),max_workers=max([cpu_count()-1,2]))
        # br1 = BioReader(file1.absolute(),max_workers=max([cpu_count()-1,2]))
        # br2 = BioReader(file2.absolute(),max_workers=max([cpu_count()-1,2]))

        br0 = BioReader(str(og.absolute()))
        br1 = BioReader(str(file1.absolute()))
        br2 = BioReader(str(file2.absolute()))


        logger.info(br1.num_x())
        logger.info(br1.num_y())
        logger.info(br1.num_z())
        logger.info(br1.num_c())
        logger.info(br1.num_t())

        logger.info(br2.num_x())
        logger.info(br2.num_y())
        logger.info(br2.num_z())
        
        image0 = br0.read_image()
        image1 = br1.read_image()
        image2 = br2.read_image()

        logger.info(np.squeeze(image1).shape)
        logger.info(np.squeeze(image2).shape)
        print("hi")
        for i in range(0, 1023):
            for j in range(0, 1023):
                # 
                if image1[i][j] != image2[i][j]:
                    # print(image1[item][item2], image2[item][item2])
                    print(i, j, ": we got a problem here -", image1[i][j], image2[i][j])
                else:
                    continue
                    # print(i,j, "IMAGE: ", image1[i][j], image2[i][j])


    finally:
        logger.info("DONE")
        logger.info('Closing the javabridge...')
        jutil.kill_vm()

