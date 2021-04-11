# Code sourced from https://github.com/MouseLand/cellpose/tree/master/cellpose

from scipy.ndimage.filters import maximum_filter1d
import scipy.ndimage
import numpy as np
from numba import njit

def diameters(masks):
    """ Get median 'diameter' of masks
    Args:
        masks(array[int]): 2D  array.labelled masks 0=NO masks; 1,2,...=mask labels
    Returns:
        md(array) :median 'diameter' of masks

    """

    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return md, counts**0.5


@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """ Run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Args:
        T(array[float64]): _ x Lx array that diffusion is run in
        y(array[int32]): pixels in y inside mask
        x(array[int32]): , pixels in x inside mask
        ymed(int32): center of mask in y
        xmed(int32): center of mask in x
        Lx(int32): size of x-dimension of masks
        niter(int32): number of iterations to run diffusion
    Returns:
        T(array[float64]):  amount of diffused particles at each pixel

    """

    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx + x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                                  T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                                  T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                                  T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])
    return T


def labels_to_flows(labels):
    """ Convert labels ( masks or flows) to flows for training model
    Args:
        labels(array):  is used to create flows and cell probabilities.
    Returns:
        flows(array): l[3 x Ly x Lx] arrays flows[1] is cell probability, flows[2] is Y flow, and flows[3] is X flow

    """

    labels=[labels]
    if labels[0].ndim < 3:
        labels = [labels[0][np.newaxis, :, :]]
    # compute flows
    veci = [masks_to_flows(labels[0][0])[0]]
    # concatenate flows with cell probability
    flows = np.concatenate((veci[0],labels[0][[0]] > 0.5), axis=0).astype(np.float32)
    return flows


def masks_to_flows(masks):
    """ Convert masks to flows using diffusion from center pixel.Center of masks where diffusion starts is defined to be
    the  closest pixel to the median of all pixels that is inside the mask. Result of diffusion is converted into flows
    by computing the gradients of the diffusion density map.
    Args:
        masks(array[int]): 2D  array.labelled masks 0=NO masks; 1,2,...=mask labels
    Returns:
        mu(array[float]): 2D array.flows in Y = mu[-2], flows in X = mu[-1].
        mu_c(array[float]): 2D array.for each pixel, the distance to the center of the mask in which it resides

    """

    if masks.ndim > 2:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows(masks[z])[0]
            mu[[1, 2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows(masks[:, y])[0]
            mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows(masks[:, :, x])[0]
            mu[[0, 1], :, :, x] += mu0
        return mu, None

    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    nmask = masks.max()
    slices = scipy.ndimage.find_objects(masks)
    dia = diameters(masks)[0]
    s2 = (.15 * dia) ** 2
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y, x = np.nonzero(masks[sr, sc] == (i + 1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x - xmed) ** 2 + (y - ymed) ** 2)
            xmed = x[imin]
            ymed = y[imin]

            d2 = (x - xmed) ** 2 + (y - ymed) ** 2
            mu_c[sr.start + y - 1, sc.start + x - 1] = np.exp(-d2 / s2)

            niter = 2 * np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly + 2) * (lx + 2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), niter)
            T[(y + 1) * lx + x + 1] = np.log(1. + T[(y + 1) * lx + x + 1])

            dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
            dx = T[y * lx + x + 1] - T[y * lx + x - 1]
            mu[:, sr.start + y - 1, sc.start + x - 1] = np.stack((dy, dx))

    mu /= (1e-20 + (mu ** 2).sum(axis=0) ** 0.5)
    return mu, mu_c

