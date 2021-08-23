import os
import cv2
import numpy as np
import logging, sys
from bfio import BioReader, BioWriter
from pathlib import Path
from aicsimageio import AICSImage
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from aicssegmentation.core.utils import get_middle_frame, hole_filling, get_3dseed_from_mid_frame
from skimage.morphology import remove_small_objects, watershed, dilation, ball


def segment_images(inpDir, outDir, config_data): 
    """ Workflow for data with shell like shapes 
    such as lamin B1 (interphase-specific)

    Args:
        inpDir : path to the input directory
        outDir : path to the output directory
        config_data : path to the configuration file
    """

    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    inpDir_files = os.listdir(inpDir)
    for i,f in enumerate(inpDir_files):
        logger.info('Segmenting image : {}'.format(f))
        
        # Load image
        br = BioReader(os.path.join(inpDir,f))
        image = br.read_image()
        structure_channel = 0 
        struct_img0 = image[:,:,:,structure_channel,0]
        struct_img0 = struct_img0.transpose(2,0,1).astype(np.float32)

        # main algorithm
        intensity_scaling_param = config_data['intensity_scaling_param']
        struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param) 
        gaussian_smoothing_sigma = config_data['gaussian_smoothing_sigma'] 
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        middle_frame_method = config_data['middle_frame_method']
        mid_z = get_middle_frame(structure_img_smooth, method=middle_frame_method)
        f2_param = config_data['f2_param']
        bw_mid_z = filament_2d_wrapper(structure_img_smooth[mid_z,:,:], f2_param)
        hole_max = config_data['hole_max']
        hole_min = config_data['hole_min']
        bw_fill_mid_z = hole_filling(bw_mid_z, hole_min, hole_max)
        seed = get_3dseed_from_mid_frame(np.logical_xor(bw_fill_mid_z, bw_mid_z), struct_img.shape, mid_z, hole_min)
        bw_filled = watershed(struct_img, seed.astype(int), watershed_line=True)>0
        seg = np.logical_xor(bw_filled, dilation(bw_filled, selem=ball(1)))
        seg = seg > 0
        out_img=seg.astype(np.uint8)
        out_img[out_img>0]=255

        # create output image
        out_img = out_img.transpose(1,2,0)
        out_img = out_img.reshape((out_img.shape[0], out_img.shape[1], out_img.shape[2], 1, 1))

        # write image using BFIO
        bw = BioWriter(os.path.join(outDir,f), metadata=br.read_metadata())
        bw.num_x(out_img.shape[1])
        bw.num_y(out_img.shape[0])
        bw.num_z(out_img.shape[2])
        bw.num_c(out_img.shape[3])
        bw.num_t(out_img.shape[4])
        bw.pixel_type(dtype='uint8')
        bw.write_image(out_img)
        bw.close_image()


