
import numpy as np
import cv2

def _taper_mask(bsize=224, sig=7.5):
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask[:, np.newaxis]
    return mask


#there
def unaugment_tiles(y):
    """ reverse test-time augmentations for averaging

    Parameters
    ----------

    y: float32
        array that's ntiles_y x ntiles_x x chan x Ly x Lx where chan = (dY, dX, cell prob);
    
    Returns
    -------

    y: float32

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j%2==0 and i%2==1:
                y[j,i] = y[j,i, :,::-1, :]
                y[j,i,0] *= -1
            elif j%2==1 and i%2==0:
                y[j,i] = y[j,i, :,:, ::-1]
                y[j,i,1] *= -1
            elif j%2==1 and i%2==1:
                y[j,i] = y[j,i, :,::-1, ::-1]
                y[j,i,0] *= -1
                y[j,i,1] *= -1
    return y


#there
def average_tiles(y, ysub, xsub, Ly, Lx):
    """ average results of network over tiles

    Parameters
    -------------

    y: float, [ntiles x 3 x bsize x bsize]
        output of cellpose network for each tile

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of pre-tiled image in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of pre-tiled image in X (may be larger than original image if
        image size is less than bsize)

    Returns
    -------------

    yf: float32, [3 x Ly x Lx]
        network output averaged over tiles

    """
    Navg = np.zeros((Ly,Lx))
    yf = np.zeros((3, Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(bsize=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf



#there
def make_tiles(imgi, bsize=224, augment=True):
    """ make tiles of image to run at test-time

    there are 4 versions of tiles
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of total image pre-tiling in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of total image pre-tiling in X (may be larger than original image if
        image size is less than bsize)

    """

    bsize = np.int32(bsize)
    nchan, Ly0, Lx0 = imgi.shape
    # pad if image smaller than bsize
    if Ly0<bsize:
        imgi = np.concatenate((imgi, np.zeros((nchan,bsize-Ly0, Lx0))), axis=1)
        Ly0 = bsize
    if Lx0<bsize:
        imgi = np.concatenate((imgi, np.zeros((nchan,Ly0, bsize-Lx0))), axis=2)
    Ly, Lx = imgi.shape[-2:]

    if augment:
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(1.75 * Ly / bsize)))
        nx = max(2, int(np.ceil(1.75 * Lx / bsize)))
        ystart = np.linspace(0, Ly-bsize, ny).astype(int)
        xstart = np.linspace(0, Lx-bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsize])
                xsub.append([xstart[i], xstart[i]+bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j%2==0 and i%2==1:
                    IMG[j,i] = IMG[j,i, :,::-1, :]
                elif j%2==1 and i%2==0:
                    IMG[j,i] = IMG[j,i, :,:, ::-1]
                elif j%2==1 and i%2==1:
                    IMG[j,i] = IMG[j,i,:, ::-1, ::-1]
    else:
        # tiles overlap by 10% tile size
        ny = 1 if Ly<=bsize else int(np.ceil(1.2 * Ly / bsize))
        nx = 1 if Lx<=bsize else int(np.ceil(1.2 * Lx / bsize))
        ystart = np.linspace(0, Ly-bsize, ny).astype(int)
        xstart = np.linspace(0, Lx-bsize, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsize])
                xsub.append([xstart[i], xstart[i]+bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
        
    return IMG, ysub, xsub, Ly, Lx

def normalize99(img):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X


#there
def reshape(data, channels=[0,0], invert=False):
    """ reshape data using channels and normalize intensities (w/ optional inversion)

    Parameters
    ----------

    data : numpy array that's (Z x ) Ly x Lx x nchan

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    invert : bool
        invert intensities

    Returns
    -------
    data : numpy array that's nchan x (Z x ) Ly x Lx

    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:,:,np.newaxis]
    elif data.shape[0]<8 and data.ndim==3:
        data = np.transpose(data, (1,2,0))

    # use grayscale image
    if data.shape[-1]==1:
        data = normalize99(data)
        if invert:
            data = -1*data + 1
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0]==0:
            data = data.mean(axis=-1)
            data = np.expand_dims(data, axis=-1)
            data = normalize99(data)
            if invert:
                data = -1*data + 1
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0]-1]
            if channels[1] > 0:
                chanid.append(channels[1]-1)
            data = data[...,chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[...,i]) > 0.0:
                    data[...,i] = normalize99(data[...,i])
                else:
                    if i==0:
                        print("WARNING: 'chan to seg' has value range of ZERO")
                    #else:
                    #    print("WARNING: 'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            
    if data.ndim==4:
        data = np.transpose(data, (3,0,1,2))
    else:
        data = np.transpose(data, (2,0,1))
    return data

def normalize_img(img):
    """ normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    Parameters
    ------------

    img: ND-array
        image of size [nchan x Ly x Lx]

    Returns
    ---------------

    img: ND-array, float32
        normalized image of size [nchan x Ly x Lx]

    """
    img = img.astype(np.float32)
    for k in range(img.shape[0]):
        if np.ptp(img[k]) > 0.0:
            img[k] = normalize99(img[k])
    return img




#there
def resize_image(img0, Ly, Lx):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [y x x x nchan] or [Lz x y x x x nchan]

    Returns
    --------------

    imgs: ND-array 
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if img0.ndim==4:
        imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly))
    else:
        imgs = cv2.resize(img0, (Lx, Ly))
    return imgs


#there

def pad_image_ND(img0, div=16, extra = 1):
    """ pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)

    Parameters
    -------------

    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]

    div: int (optional, default 16)

    Returns
    --------------

    I: ND-array
        padded image

    ysub: array, int
        yrange of pixels in I corresponding to img0

    xsub: array, int
        xrange of pixels in I corresponding to img0

    """
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1+Ly)
    xsub = np.arange(ypad1, ypad1+Lx)
    return I, ysub, xsub



def _X2zoom(img, X2=1):
    """ zoom in image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    Returns
    -------
    img : numpy array that's Ly x Lx

    """
    ny,nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2**X2)), int(ny * (2**X2))))
    return img

def _image_resizer(img, resize=512, to_uint8=False):
    """ resize image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    resize : int
        max size of image returned

    to_uint8 : bool
        convert image to uint8

    Returns
    -------
    img : numpy array that's Ly x Lx, Ly,Lx<resize

    """
    ny,nx = img.shape[:2]
    if to_uint8:
        if img.max()<=255 and img.min()>=0 and img.max()>1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img
