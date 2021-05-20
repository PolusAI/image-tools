from bfio import  BioWriter , OmeXml
import argparse, logging, sys
import numpy as np
from pathlib import Path
import zarr
import mask
import scipy.ndimage
from numba import jit
np.seterr(divide='ignore', invalid='ignore')

# @jit(nopython=True)
# def _label_overlap(x, y):
#     """ Fast function to get pixel overlaps between masks in x and y
#     Args:
#         x(array[int]): ND-array where 0=NO masks; 1,2... are mask labels
#         y(array[int]): ND-array where 0=NO masks; 1,2... are mask labels
#
#     Returns:
#         overlap(array[int]): ND-array.matrix of pixel overlaps of size [x.max()+1, y.max()+1]
#
#     """
#
#     x = x.ravel()
#     y = y.ravel()
#     overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
#     for i in range(len(x)):
#         overlap[x[i], y[i]] += 1
#     return overlap
#
# def _intersection_over_union(masks_true, masks_pred):
#     """Intersection over union of all mask pairs
#     Args:
#         masks_true(array[int]): ND-array.ground truth masks, where 0=NO masks; 1,2... are mask labels
#         masks_pred(array[int]): ND-array.predicted masks, where 0=NO masks; 1,2... are mask labels
#
#     Returns:
#         iou(array[float]): ND-array.matrix of IOU pairs of size [x.max()+1, y.max()+1]
#
#     """
#
#     overlap = _label_overlap(masks_true, masks_pred)
#     n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
#     n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
#
#     iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
#     iou[np.isnan(iou)] = 0.0
#     return iou
#
# def stitch3D(masks,tile_size, stitch_threshold=0.25):
#     """ Stitch 2D masks into 3D volume with stitch_threshold on IOU
#     Args:
#         masks(array):ND-array.Mask labels
#         stitch_threshold(int): Stitching threshold.Default value of 0.25
#
#     Returns:
#         masks(array): Stitched masks based on IOU
#
#     """
#     mmax = masks[0].max()
#
#     for z in range(masks.shape[2] - 1):
#         for x in range(0, masks.shape[1], tile_size):
#             x_max = min([masks.shape[1], x + tile_size])
#             for y in range(0, masks.shape[0], tile_size):
#                 y_max = min([masks.shape[0], y + tile_size])
#                 iou = _intersection_over_union(np.array(masks[y:y_max, x:x_max, z + 1].squeeze()),
#                                                np.array(masks[y:y_max, x:x_max, z].squeeze()))[1:, 1:]
#                 iou[iou < stitch_threshold] = 0.0
#
#                 if not (iou.shape[0] == 0 or iou.shape[1] == 0):
#                     iou[iou < iou.max(axis=0)] = 0.0
#                     istitch = iou.argmax(axis=1) + 1
#                     ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
#                     istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
#                     mmax += len(ino)
#                     istitch = np.append(np.array(0), istitch)
#                     stitched = istitch[masks[y:y_max, x:x_max, z + 1].squeeze()]
#                     masks[y:y_max, x:x_max, z + 1] = stitched[:, :]
#
#     return masks
# for l in range(1, n):
#     if objects[l] is not None:
#         # Get individual bounding box
#         bb_slices = objects[l]
#         # Extract object
#         obj = labels[bb_slices]
#         # Compute centroid
#
#         coord = scipy.ndimage.center_of_mass(obj)
#         y_test, x_test = np.unravel_index(np.argmax(obj), obj.shape)
#         print(y_test, x_test)
#         # Translate result from coordinate space of the bounding box back to the source image by simply adding the starting coordinate of each slice
#         #    coord_translated = (np.around(coord[0]).astype(np.uint8) + bb_slices[0].start ,
#         #                       np.around(coord[1]).astype(np.uint8) + bb_slices[1].start)
#
#         coord_translated = (x_test + bb_slices[0].start,
#                             y_test + bb_slices[1].start)
#         print(coord_translated)
#         if x_test == 1024 or y_test == 1024:
#             coords.append(obj)
def centre(labels,x,y):
    coords = []
    n = labels.max()
    # Get bounding boxes for all objects in the form of slices
    objects = scipy.ndimage.find_objects(labels)
    # Loop over all objects:
    import cv2
    for i, si in enumerate(objects):
        if si is not None:
            sr, sc = si
            mask = (labels[sr, sc] == (i + 1)).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
            vr, vc = pvr + sr.start, pvc + sc.start
            for a in (vr,vc):

                if a[0] == 1024 or a[1]== 1024:
                    print(a)
                    coords.append(mask)
                    print(mask)
    return coords

TILE_SIZE = 2048
TILE_OVERLAP = 512

#Counter for masks across tiles
total_pix=0

def set_totalpix(a):
    """Counter for  number of masks predicted
    Args:
        a(int):  Number of maks in a tile
    """
    global total_pix
    if a!=0:
        total_pix=a
    else:
        total_pix=0

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
    # parser.add_argument('--stitchThreshold',required=False, type=float,
    #                     help='Stitch threshold for 3D', default=0.0)
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
   # stitch_threshold=args.stitchThreshold
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

            path = Path(outDir).joinpath(str(file_name))
            xml_metadata = OmeXml.OMEXML(metadata)

            with  BioWriter(file_path=Path(path), backend='python', metadata=xml_metadata) as bw:
                bw.dtype=np.dtype(np.uint32)

                # Iterating over Z dimension
                for z in range(0, root[file_name]['vector'].shape[2], 1):
                    new_img = -1
                    global total_pix
                    set_totalpix(0)

                    for x in range(0, root[file_name]['vector'].shape[1], TILE_SIZE):

                        for y in range(0, root[file_name]['vector'].shape[0], TILE_SIZE):
                            x_min = max([0, x - TILE_OVERLAP])
                            x_max = min([root[file_name]['vector'].shape[1], x + TILE_SIZE + TILE_OVERLAP])

                            y_min = max([0, y - TILE_OVERLAP])
                            y_max = min([root[file_name]['vector'].shape[1], y + TILE_SIZE + TILE_OVERLAP])


                            tile = root[file_name]['vector'][y_min:y_max, x_min:x_max, z:z + 1, :, :]
                            tile=tile.transpose((2, 0, 1, 3, 4)).squeeze()
                            tile_final= tile

                            logger.info('Calculating flows and masks  for tile [{}:{},{}:{},{}:{}]'.format(y, y_max, x,
                                        x_max, z, z + 1))

                            cellprob = tile_final[..., -1]
                            dP = np.stack((tile_final[..., 0], tile_final[..., 1]), axis=0)

                            # Computing flows for the tile

                            p = mask.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                              niter=niter, interp=True)

                            # Generating masks for the tile
                            maski = mask.compute_masks(p, cellprob, dP,total_pix, cellprob_threshold,
                                                            flow_threshold)

                        #    coords = centre(maski,x_min , y_min)



                            # reshaping mask  based on tile
                            x_overlap = x - x_min
                            x_min = x
                            x_max = min([root[file_name]['vector'].shape[1], x + TILE_SIZE])

                            y_overlap = y - y_min
                            y_min = y
                            y_max = min([root[file_name]['vector'].shape[0], y + TILE_SIZE])

              #              print(coords)
                            maski=maski[:,:, np.newaxis].astype(np.uint32)

                            test=maski[y_overlap:y_max - y_min + y_overlap,x_overlap:x_max - x_min + x_overlap, :, np.newaxis,
                                                                  np.newaxis]

                            bw[y_min:y_max, x_min:x_max, z:z + 1, 0, 0] = test
                            cnt = np.amax(test) if np.amax(test) != 0 else total_pix
                            set_totalpix(cnt)

                            new_img = 1

  #              del maski,old_tile,new_tile,mask_final
            count += 1
    finally:
        logger.info('Closing ')
        # Exit the program
        sys.exit()

if __name__ == '__main__':
    main()