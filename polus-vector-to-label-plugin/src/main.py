from bfio import  BioWriter , OmeXml
import argparse, logging, sys
import numpy as np
from pathlib import Path
import zarr
import mask
from numba import jit
np.seterr(divide='ignore', invalid='ignore')

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

    for z in range(masks.shape[2] - 1):
        for x in range(0, masks.shape[1], tile_size):
            x_max = min([masks.shape[1], x + tile_size])
            for y in range(0, masks.shape[0], tile_size):
                y_max = min([masks.shape[0], y + tile_size])
                iou = _intersection_over_union(np.array(masks[y:y_max, x:x_max, z + 1].squeeze()),
                                               np.array(masks[y:y_max, x:x_max, z].squeeze()))[1:, 1:]
                iou[iou < stitch_threshold] = 0.0

                if not (iou.shape[0] == 0 or iou.shape[1] == 0):
                    iou[iou < iou.max(axis=0)] = 0.0
                    istitch = iou.argmax(axis=1) + 1
                    ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
                    istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
                    mmax += len(ino)
                    istitch = np.append(np.array(0), istitch)
                    stitched = istitch[masks[y:y_max, x:x_max, z + 1].squeeze()]
                    masks[y:y_max, x:x_max, z + 1] = stitched[:, :]

    return masks

def crop_center(img,cropx=1024,cropy=1024):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def main():
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
        # Open zarr file
        root = zarr.open(str(Path(inpDir).joinpath('flow.zarr')),mode='r')
        count=0

        # Loop through files in inpDir image collection and process
        for file_name, vec in root.groups():
            logger.info(
                'Processing image ({}/{}): {}'.format(count + 1, len([file_name for file_name, _ in root.groups()]),
                                                      file_name))
            metadata = vec.attrs['metadata']
            tile_size = min(1024, root[file_name]['vector'].shape[1])
            path = Path(outDir).joinpath(str(file_name))
            xml_metadata = OmeXml.OMEXML(metadata)

            with  BioWriter(file_path=Path(path), backend='python', metadata=xml_metadata) as bw:
                bw.dtype=np.dtype(np.uint32)

                # Iterating over Z dimension
                for z in range(0, root[file_name]['vector'].shape[2], 1):
                    new_img = -1
                    mask_final = np.zeros((root[file_name]['vector'].shape[0], root[file_name]['vector'].shape[1],
                                           1)).astype(np.uint32)
                    for y in range(0, root[file_name]['vector'].shape[1], tile_size):
                        y_max = min([root[file_name]['vector'].shape[1], y + tile_size])

                        for x in range(0, root[file_name]['vector'].shape[1], tile_size):

                            x_max = min([root[file_name]['vector'].shape[1], x + tile_size])
                            tile = root[file_name]['vector'][y:y_max, x:x_max, z:z + 1, :, :]
                            tile=tile.transpose((2, 0, 1, 3, 4)).squeeze()
                            tile_final= tile
                            # adding last column of previous tile along row
                            if x!=0:

                                 tile_final=np.concatenate((x_cache,tile_final),axis=1)
                            #  adding row of the  previous tile along column
                            if y!=0:
                                if x!=0:
                                    y_tile = root[file_name]['vector'][y-1:y, x-1:x_max, z:z + 1, :, :].squeeze()

                                else:
                                    y_tile = root[file_name]['vector'][y - 1:y, x:x_max, z:z + 1, :, :].squeeze()

                                y_tile = y_tile[np.newaxis,:,:]
                                tile_final=np.concatenate((y_tile,tile_final),axis=0)

                            logger.info('Calculating flows and masks  for tile [{}:{},{}:{},{}:{}]'.format(y, y_max, x,
                                        x_max, z, z + 1))

                            cellprob = tile_final[..., -1]
                            dP = np.stack((tile_final[..., 0], tile_final[..., 1]), axis=0)

                            # Computing flows for the tile

                            p = mask.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                              niter=niter, interp=True)

                            # Generating masks for the tile
                            maski = mask.compute_masks(p, cellprob, dP, new_img,z, cellprob_threshold,
                                                            flow_threshold)
                            # reshaping mask  based on tile
                            if x != 0:
                                maski=maski[:,:-1]

                            if y!= 0:
                                maski=maski[:-1,:]

                            mask_final[y:y_max, x:x_max,0:1]=maski[:,:, np.newaxis].astype(np.uint32)
                            new_img = 1
                            x_cache = tile[:, -1, :3]
                            x_cache = x_cache[:,np.newaxis,:]

                    new_tile=mask_final
                    if z > 0 and stitch_threshold > 0 :
                        logger.info('stitching   masks into 3D volume for slice {}:{}'.format( z,z + 1))
                        tilez_stack=np.stack((old_tile[:, :, :],new_tile[:, :, :]),axis=2)
                        mask_3d=stitch3D(tilez_stack.squeeze(),tile_size,stitch_threshold=stitch_threshold)
                        bw[0:y_max,0:x_max, z-1:z+1, 0, 0] = mask_3d[:, :,:, np.newaxis,
                                                                      np.newaxis].astype(np.uint32)

                    else:
                        if not stitch_threshold >0 or root[file_name]['vector'].shape[2] ==1:
                            bw[0:y_max, 0:x_max, z:z + 1, 0, 0] = mask_final[:, :, :, np.newaxis,
                                                                          np.newaxis].astype(np.uint32)

                    old_tile=mask_final

                del maski,old_tile,new_tile,mask_final
            count += 1
    finally:
        logger.info('Closing ')
        # Exit the program
        sys.exit()

if __name__ == '__main__':
    main()