from bfio import  BioWriter , OmeXml
import argparse, logging, sys
import numpy as np
from pathlib import Path
import zarr
import mask
from numba import jit


np.seterr(divide='ignore', invalid='ignore')

lbl_dtype={'uint8':255,'uint16':65535,'uint32':4294967295,'float':3.402823E+38}



@jit(nopython=True)
def _label_overlap(x, y):
    """ Fast function to get pixel overlaps between masks in x and y
    Args:
        x(array[int]): ND-array where 0=NO masks; 1,2... are mask labels
        y(array[int]): ND-array where 0=NO masks; 1,2... are mask labels

    Returns:
        overlap(array[int]): ND-array.matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """

    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """Intersection over union of all mask pairs
    Args:
        masks_true(array[int]): ND-array.ground truth masks, where 0=NO masks; 1,2... are mask labels
        masks_pred(array[int]): ND-array.predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns:
        iou(array[float]): ND-array.matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """

    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def stitch3D(masks,tile_size, stitch_threshold=0.25):
    """ Stitch 2D masks into 3D volume with stitch_threshold on IOU
    Args:
        masks(array):ND-array.Mask labels
        stitch_threshold(int): Stitching threshold.Default value of 0.25

    Returns:
        masks(array): Stitched masks based on IOU

    """
    mmax = masks[0].max()

    for z in range(masks.shape[2]-1):
        for x in range(0, masks.shape[1], tile_size):
            x_max = min([masks.shape[1], x + tile_size])
            for y in range(0, masks.shape[0], tile_size):
                y_max = min([masks.shape[0], y + tile_size])
                iou = _intersection_over_union(np.array(masks[y:y_max, x:x_max ,z+1].squeeze()),np.array(masks[y:y_max, x:x_max ,z].squeeze()))[1:,1:]
                iou[iou < stitch_threshold] = 0.0

                if not (iou.shape[0] == 0 or iou.shape[1] == 0):
                    iou[iou < iou.max(axis=0)] = 0.0
                    istitch = iou.argmax(axis=1) + 1
                    ino = np.nonzero(iou.max(axis=1)==0.0)[0]
                    istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
                    mmax += len(ino)
                    istitch = np.append(np.array(0), istitch)
                    stitched = istitch[masks[y:y_max, x:x_max ,z+1].squeeze()]
                    masks[y:y_max, x:x_max ,z+1] = stitched[:,:]

    return masks


def main():
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')
    
    # Input arguments counter=0

    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--flowThreshold', required=False,
                        default=0.8, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--cellprobThreshold', required=False,
                        default=0.0, type=float, help='cell probability threshold, centered at 0.0')
    parser.add_argument('--stitchThreshold',required=False, type=float,
                        help='Stitch threshold for 3D', default=0.0)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    cellprob_threshold = args.cellprobThreshold
    flow_threshold= args.flowThreshold
    stitch_threshold=args.stitchThreshold
    rescale = np.ones(1)
    niter = 1 / rescale[0] * 200
    # Surround with try/finally for proper error catching
    try:
        logger.info('Initializing ...')
        # Get all file names in inpDir image collection
        root = zarr.open(str(Path(inpDir).joinpath('flow.zarr')),mode='r')
        count=0

        # Loop through files in inpDir image collection and process
        for file_name,vec in root.groups():
            logger.info('Processing image ({}/{}): {}'.format(count + 1, len([file_name for file_name, _ in root.groups()]), file_name))
            vec_arr = vec['vector']
            vec_arr = np.asarray(vec_arr)
            metadata = vec.attrs['metadata']
            mask_final = np.zeros((vec_arr.shape[0],vec_arr.shape[1],vec_arr.shape[2],1,1))
            vec_arr = vec_arr.transpose((2,0,1,3,4)).squeeze(axis=4)
            tile_size = min(1024,vec_arr.shape[1])
            lblcnt_max=0
            # Iterating over Z dimension

            for z in range(vec_arr.shape[0]):
                    logger.info('Calculating flows for slice {} of image {}'.format(z+1, file_name))
                    new_img = -1
                    for x in range(0, vec_arr.shape[2], tile_size):
                        x_max = min([vec_arr.shape[2], x + tile_size])
                        for y in range(0, vec_arr.shape[1], tile_size):
                            y_max = min([vec_arr.shape[1], y + tile_size])
                            prob = vec_arr[z,y:y_max, x:x_max , :].astype(np.float32)
                            cellprob = prob[..., -1]
                            dP = np.stack((prob[..., 0], prob[..., 1]), axis=0)
                            # Computing flows for the tile
                            p = mask.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                                                    niter=niter, interp=True)
                            # Generating masks for the tile
                            maski,lbl_cnt = mask.compute_masks(p,cellprob,dP,new_img,cellprob_threshold,flow_threshold)
                            mask_final=mask_final.astype(maski.dtype)
                            mask_final[y:y_max, x:x_max,z:z+1,:,:] = maski[:,:,np.newaxis,np.newaxis,np.newaxis].astype(maski.dtype)
                            new_img = 1
                            lblcnt_max= max(lbl_cnt,lblcnt_max)

            logger.info('Computed  masks for  image {}'.format(file_name))

            if mask_final.shape[2] > 1 and stitch_threshold > 0:
                logger.info('stitching   masks into 3D volume for image {}'.format(file_name))
                mask_final = stitch3D(mask_final.squeeze(),tile_size, stitch_threshold=stitch_threshold)
                mask_final = mask_final[...,np.newaxis,np.newaxis]

            # Setting final array dtype based on number of labels.
            for i, (key, value) in enumerate(lbl_dtype.items()):
                if lblcnt_max < value:
                    break

            mask_final=np.array(mask_final, dtype=key)
            xml_metadata = OmeXml.OMEXML(metadata)
            # Write the output
            logger.info('Saving label for image {}'.format(file_name))
            path=Path(outDir).joinpath(str(file_name))
            bw = BioWriter(file_path=Path(path),backend='python',metadata=xml_metadata)
            bw.dtype=mask_final.dtype
            bw.write(mask_final)
            bw.close()
            del mask_final,bw
            count+=1
    finally:
        logger.info('Closing ')
        # Exit the program
        sys.exit()

if __name__ == '__main__':
    main()