import logging
import os
from multiprocessing import cpu_count

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT', '.ome.tif')

NUM_THREADS = max(1, int(cpu_count() * 0.8))
TILE_STRIDE = 1024 * 2  # TODO: Measure impact of making this larger

"""
The number of bins in each distogram. 32-128 are reasonable values and
have no noticeable difference in performance. More than 128 bins starts to slow
the computations.
"""
MAX_BINS = 32

"""
Whether to adaptively resize bins based on density of pixel values. Using this
offers decent improvement in entropy calculations at minimal added runtime cost.
"""
WEIGHTED_BINS = True

"""
The number of rows/columns over which to compute a rolling mean during entropy
and gradient calculations. Setting this to larger values decreases sensitivity
to noise but comes with the risk of confusing tissue with noise. Due to how the
entropy and its gradient are calculated, this value also serves as the number
of rows/columns to pad around a bounding box.
"""
WINDOW_SIZE = 32

"""
If the entropy gradient across rows/columns is ever greater than this value,
stop the search and mark that location as a cropping boundary. Setting this
value too low increases sensitivity to noise but also avoids cropping away
actual tissue.
"""
GRADIENT_THRESHOLD = 1e-2

"""
If we do not hit the gradient threshold in a reasonable number of rows/columns,
we simply use the location of this percentile gradient as the cutoff.
"""
GRADIENT_PERCENTILE = 90.0
