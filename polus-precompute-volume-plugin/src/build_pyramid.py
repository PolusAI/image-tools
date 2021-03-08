from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import logging, argparse, traceback
from bfio.bfio import BioReader, BioWriter

import utils
import os
import tempfile

import numpy as np

from neurogen import info as nginfo
from neurogen import volume as ngvol


if __name__=="__main__":
    try:
        # Setup the Argument parsing
        parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

        parser.add_argument('--inpDir', dest='input_dir', type=str,
                            help='Path to folder with CZI files', required=True)
        parser.add_argument('--outDir', dest='output_dir', type=str,
                            help='The output directory for ome.tif files', required=True)
        parser.add_argument('--imagetype', dest='image_type', type=str,
                            help='The type of image: image or segmentation', required=True)
        parser.add_argument('--mesh', dest='mesh', type=bool, nargs='?',const=True,
                        default=False,help='True or False', required=True)

        # Get arguments
        args = parser.parse_args()
        input_image = args.input_dir
        output_dir = args.output_dir
        imagetype = args.image_type
        mesh = args.mesh
        image = os.path.basename(input_image)

        # Initialize Logger
        logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("build_pyramid")
        logger.setLevel(logging.INFO) 

        logger.info("Starting to build...")

        # Create the BioReader object
        logger.info('Getting the BioReader...')

        bf = BioReader(input_image, max_workers=max([cpu_count()-1,2]))
        bfshape = bf.shape
        logger.info("Image Shape {}".format(bfshape))
        datatype = np.dtype(bf.dtype)
        chunk_size = [256,256,256]
        bit_depth = 16
        resolution = utils.get_resolution(phys_y=bf.physical_size_y, 
                                          phys_x=bf.physical_size_x, 
                                          phys_z=bf.physical_size_z)

        if imagetype == "segmentation":
            if mesh == False:
                file_info = nginfo.info_segmentation(directory=output_dir,
                                                    dtype=datatype,
                                                    chunk_size = chunk_size,
                                                    size=bfshape[:3],
                                                    resolution=resolution)
                encodedvolume = ngvol.generate_recursive_chunked_representation(volume=bf,info=file_info, dtype=datatype, directory=output_dir)

            else:
                ysplits = list(np.arange(0, bfshape[0], chunk_size[0]))
                ysplits.append(bfshape[0])
                xsplits = list(np.arange(0, bfshape[1], chunk_size[1]))
                xsplits.append(bfshape[1])
                zsplits = list(np.arange(0, bfshape[2], chunk_size[2]))
                zsplits.append(bfshape[2])

                all_identities = np.array([])
                totalbytes = {}
                temp_dir = os.path.join(output_dir, "tempdir")

                with tempfile.TemporaryDirectory() as temp_dir:
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir, exist_ok=True)
                    for y in range(len(ysplits)-1):
                        for x in range(len(xsplits)-1):
                            for z in range(len(zsplits)-1):
                                start_y, end_y = (ysplits[y], ysplits[y+1])
                                start_x, end_x = (xsplits[x], xsplits[x+1])
                                start_z, end_z = (zsplits[z], zsplits[z+1])
                                

                                volume = bf[start_y:end_y,start_x:end_x,start_z:end_z]
                                volume = volume.reshape(volume.shape[:3])
                                logger.info("Loaded subvolume (YXZ) {}-{}__{}-{}__{}-{}".format(start_y, end_y,
                                                                                                start_x, end_x,
                                                                                                start_z, end_z))
                                ids = np.unique(volume)
                                if (ids == [0]).all():
                                    continue
                                else:
                                    ids = np.delete(ids, np.where(ids==0))
                                    with ThreadPoolExecutor(max_workers=8) as executor:
                                        executor.submit(utils.create_plyfiles(subvolume = volume,
                                                                        ids=ids,
                                                                        temp_dir=temp_dir,
                                                                        start_y=start_y,
                                                                        start_x=start_x,
                                                                        start_z=start_z,
                                                                        totalbytes=totalbytes))
                                        all_identities = np.append(all_identities, ids)

                    executor.shutdown(wait=True)

                    all_identities = np.unique(all_identities).astype('int')
                    file_info = nginfo.info_mesh(directory=output_dir,
                                                chunk_size=chunk_size,
                                                size=bf.shape[:3],
                                                dtype=np.dtype(bf.dtype).name,
                                                ids=all_identities,
                                                resolution=resolution,
                                                segmentation_subdirectory="segment_properties",
                                                bit_depth=bit_depth,
                                                order="YXZ")
                    

                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futuresvariable = [executor.submit(utils.concatenate_and_generate_meshes, 
                                                        ide, temp_dir, output_dir, bit_depth, chunk_size) 
                                                        for ide in all_identities]
                    executor.shutdown(wait=True)

                    encodedvolume = ngvol.generate_recursive_chunked_representation(volume=bf,info=file_info, dtype=datatype, directory=output_dir)
        elif imagetype == "image":
            file_info = nginfo.info_image(directory=output_dir,
                                                dtype=datatype,
                                                chunk_size = [256,256,256],
                                                size=bfshape[:3],
                                                resolution=resolution)
            encodedvolume = ngvol.generate_recursive_chunked_representation(volume=bf, info=file_info, dtype=datatype, directory=output_dir, blurring_method='average')
        else:
            raise ValueError("Image Type was not properly specified")
        
        logger.info("Data Type: {}".format(file_info['data_type']))
        logger.info("Number of Channels: {}".format(file_info['num_channels']))
        logger.info("Number of Scales: {}".format(len(file_info['scales'])))
        logger.info("Image Type: {}".format(file_info['type']))

    except Exception as e:
        traceback.print_exc()


