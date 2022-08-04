import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

import numpy as np
from bfio import BioReader, BioWriter
from filepattern import FilePattern, get_regex, infer_pattern, output_name
from basicpy import BaSiC
from preadator import ProcessManager

logging.getLogger("basicpy.basicpy").setLevel(logging.WARNING)

""" Load files and create an image stack """


def _get_resized_image_stack(flist):
    """Load all images in a list and resize to OPTIONS['size']

    When files are parsed, the variables are used in an index to provide a
    method to reference a specific file name by its dimensions. This function
    returns the variable index based on the input filename pattern.

    To help reduce memory overhead, the number of images loaded for fitting are
    limited to OPTIONS['n_sample'] number of images. If there are more than
    n_sample number of images, images are selected at random.

    Inputs:
        flist - Paths of list of images to load and resize
    Outputs:
        img_stack - A 3D stack of 2D images
        X - width of image
        Y - height of image
    """

    # Initialize the output
    with BioReader(flist[0]["file"]) as br:
        X = br.x
        Y = br.y
        dtype = br.dtype

    if len(flist) > 1024:
        N = 1024
        samples = np.random.permutation(len(flist)).tolist()
        flist = [flist[s] for s in samples[:1024]]
    else:
        N = len(flist)

    img_stack = np.zeros((N, Y, X), dtype=dtype)

    def load_and_store(fname, ind):
        with ProcessManager.thread() as active_threads:
            with BioReader(fname["file"], max_workers=active_threads.count) as br:
                I = np.squeeze(br[:, :, 0, 0, 0])
            img_stack[ind, :, :] = I

    # Load every image as a z-slice
    for ind, fname in enumerate(flist):
        ProcessManager.submit_thread(load_and_store, fname, ind)

    ProcessManager.join_threads()

    return img_stack


def basic(
    files: typing.List[Path],
    out_dir: Path,
    metadata_dir: typing.Optional[Path] = None,
    get_darkfield: bool = False,
    photobleach: bool = False,
    extension: str = ".ome.tif",
):

    # Try to infer a filename
    try:
        pattern = infer_pattern([f["file"].name for f in files])
        fp = FilePattern(files[0]["file"].parent, pattern)
        base_output = fp.output_name()

    # Fallback to the first filename
    except:
        base_output = files[0]["file"].name

    with ProcessManager.process(base_output):

        # Load files and sort
        ProcessManager.log("Loading and sorting images...")
        img_stk = _get_resized_image_stack(files)

        # Run basic fit
        ProcessManager.log("Beginning flatfield estimation")
        basic = BaSiC(get_darkfield=get_darkfield)
        basic.fit(img_stk)

        # Resize images back to original image size
        ProcessManager.log("Saving outputs...")
        flatfield = basic.flatfield
        if get_darkfield:
            darkfield = basic.darkfield

        # Export the flatfield image as a tiled tiff
        flatfield_out = base_output.replace(extension, "_flatfield" + extension)

        with BioReader(files[0]["file"], max_workers=2) as br:
            metadata = br.metadata

        with BioWriter(
            out_dir.joinpath(flatfield_out), metadata=metadata, max_workers=2
        ) as bw:
            bw.dtype = np.float32
            bw[:] = flatfield

        # Export the darkfield image as a tiled tiff
        if get_darkfield:
            darkfield_out = base_output.replace(extension, "_darkfield" + extension)
            with BioWriter(
                out_dir.joinpath(darkfield_out), metadata=metadata, max_workers=2
            ) as bw:
                bw.dtype = np.float32
                bw[:] = darkfield

        # # Export the photobleaching components as csv
        # if photobleach:
        #     offsets_out = base_output.replace(extension, "_offsets.csv")
        #     with open(metadata_dir.joinpath(offsets_out), "w") as fw:
        #         fw.write("file,offset\n")
        #         for f, o in zip(files, pb[0, :].tolist()):
        #             fw.write("{},{}\n".format(f, o))
