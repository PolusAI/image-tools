from bfio import  BioWriter , OmeXml
import argparse, logging, sys
import numpy as np
from pathlib import Path
import re
import zarr
import mask
from numba import jit


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y
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
    """ intersection over union of all mask pairs
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

def stitch3D(masks, stitch_threshold=0.25):
        """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
        mmax = masks[0].max()

        for i in range(len(masks)-1):

            iou = _intersection_over_union(masks[i+1], masks[i])[1:,1:]
            iou[iou < stitch_threshold] = 0.0

            if not (iou.shape[0] ==0 or iou.shape[1]==0):
                iou[iou < iou.max(axis=0)] = 0.0
                istitch = iou.argmax(axis=1) + 1
                ino = np.nonzero(iou.max(axis=1)==0.0)[0]
                istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
                mmax += len(ino)
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]

        return masks


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
                        default=0.8, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--cellprob_threshold', required=False,
                        default=0.0, type=float, help='cell probability threshold, centered at 0.0')
    parser.add_argument('--stitch_threshold',required=False, type=float,
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
    cellprob_threshold = args.cellprob_threshold
    flow_threshold= args.flow_threshold
    stitch_threshold=args.stitch_threshold
    rescale = np.ones(1)
    niter = 1 / rescale[0] * 200
    # Surround with try/finally for proper error catching
    try:
        logger.info('Initializing ...')
        # Get all file names in inpDir image collection
        root = zarr.open(str(Path(inpDir).joinpath('flow.zarr')),mode='r')
        count=0
        # Loop through files in inpDir image collection and process
        for m,l in root.groups():
            logger.info('Processing image ({}/{}): {}'.format(count + 1, len([m for m, _ in root.groups()]), m))
            y=l['vector']
            y= np.asarray(y)
            metadata=l.attrs['metadata']
            mask_final = np.zeros((y.shape[0],y.shape[1],y.shape[2],1,1)).astype(y.dtype)
            y=y.transpose((2,0,1,3,4)).squeeze(axis=4)

            for z in range(y.shape[0]):
                prob = y[z, :, :, :].astype(np.float32)
                cellprob = prob[..., -1]
                dP = np.stack((prob[..., 0], prob[..., 1]), axis=0)
                p = mask.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                                                    niter=niter, interp=True)
                maski = mask.compute_masks(p,cellprob,dP,cellprob_threshold,flow_threshold)
                mask_final[:,:,z:z+1,:,:]=maski[:,:,np.newaxis,np.newaxis,np.newaxis].astype(maski.dtype)

            # Write the output
            temp = re.sub(r'(?<=Type=")([^">>]+)', str(mask_final.dtype), metadata)
            xml_metadata = OmeXml.OMEXML(metadata)
            path=Path(outDir).joinpath(str(m))
            bw = BioWriter(file_path=Path(path),backend='python',metadata=xml_metadata)
            bw.write(mask_final)
            bw.close()
            del mask_final
            count+=1
    finally:
        logger.info('Closing ')
        # Exit the program
        sys.exit()