"""Image dimension stacking package."""
import logging
import time
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from pathlib import Path

import filepattern as fp
import preadator
from bfio import BioReader
from bfio import BioWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

chunk_size = 1024

num_workers = max([cpu_count(), 2])


# Units for conversion
UNITS = {
    "m": 10**9,
    "cm": 10**7,
    "mm": 10**6,
    "µm": 10**3,
    "nm": 1,
    "Å": 10**-1,
}


def z_distance(file: Path) -> tuple[float, str]:
    """Get physical z-distance.

    This estimates zdistance if not provided by averaging physical distances of x and y.

    Args:
        file : Path to input image file
    Returns:
        A tuple of float and string values.
    """
    # Get some basic info about the files to stack
    with BioReader(file) as br:
        # Get the physical z-distance if available, set to physical x if not
        ps_z = br.ps_z

        # If the z-distances are undefined, average the x and y together
        if None in ps_z:
            # Get the size and units for x and y
            x_val, xunits = br.ps_x
            y_val, yunits = br.ps_y

            x_units = xunits.value
            y_units = yunits.value

            # Convert x and y values to the same units and average
            z_val = (x_val * UNITS[x_units] + y_val * UNITS[y_units]) / 2

            # Set z units to the smaller of the units between x and y
            z_units = x_units if UNITS[x_units] < UNITS[y_units] else y_units

            # Convert z to the proper unit scale
            z_val /= UNITS[z_units]
            ps_z = (z_val, z_units)

            if not ps_z:
                msg = f"Unable to find physical z-size {ps_z}"
                raise ValueError(
                    msg,
                )

        return ps_z


def write_image_stack(file: Path, di: int, group_by: str, bw: BioWriter) -> None:
    """Write image stack.

    This function writes stacked images of either dimensions (z, c, t).

    Args:
        file : Path to input image file
        di : Index of dimension
        group_by : A single string variable to group filenames by
        bw : bfio.BioWriter.

    """
    with BioReader(file, max_workers=num_workers) as br:
        for t in range(br.T):
            for c in range(br.C):
                for z in range(br.Z):
                    for y in range(0, br.Y, chunk_size):
                        y_max = min([br.Y, y + chunk_size])
                        for x in range(0, br.X, chunk_size):
                            x_max = min([br.X, x + chunk_size])
                            if group_by == "c":
                                tile = br[y:y_max, x:x_max, 0, c : c + 1, 0]

                                bw[y:y_max, x:x_max, 0, di : di + 1, 0] = tile
                            if group_by == "t":
                                tile = br[y:y_max, x:x_max, 0, t : t + 1, 0]
                                bw[y:y_max, x:x_max, 0, 0, di : di + 1] = tile

                            if group_by == "z":
                                tile = br[y:y_max, x:x_max, z : z + 1, 0, 0]
                                bw[y:y_max, x:x_max, di : di + 1, 0, 0] = tile


def dimension_stacking(
    inp_dir: Path,
    file_pattern: str,
    group_by: str,
    out_dir: Path,
) -> None:
    """Image dimension stacking.

    This function enables to write stack image of dimensions (z, c, t).
        inp_dir : Path to input directory containing images
        file_pattern : Pattern to parse image files
        group_by : A single string variable to group filenames by
        out_dir : Path to output directory.

    """
    dimensions = []
    input_files = []

    fps = fp.FilePattern(inp_dir, file_pattern)
    out_name = fps.output_name()

    for fl in fps(group_by=group_by):
        f1, f2 = fl
        if f1[0][0] == group_by:
            file_dim = f1[0][1]
            file = f2[0][1][0]
            input_files.append(file)
            dimensions.append(file_dim)

    # Get the number of layers to stack
    dim_size = len(dimensions)

    with BioReader(input_files[0]) as br:
        metadata = br.metadata

    with BioWriter(
        out_dir.joinpath(out_name),
        metadata=metadata,
        max_workers=num_workers,
    ) as bw:
        # Adjust the dimensions before writing
        if group_by == "c":
            bw.C = dim_size
        if group_by == "t":
            bw.T = dim_size
        if group_by == "z":
            bw.Z = dim_size
            bw.ps_z = z_distance(input_files[0])

        starttime = time.time()

        with preadator.ProcessManager(
            name=f"Stacking images of {group_by} dimensions",
            num_processes=num_workers,
            threads_per_process=4,
        ) as pm:
            threads = []
            for file, di in zip(input_files, range(0, dim_size)):
                thread = pm.submit_thread(
                    write_image_stack,
                    file,
                    di=di,
                    group_by=group_by,
                    bw=bw,
                )
                threads.append(thread)
            pm.join_threads()

            for f in tqdm(
                as_completed(threads),
                total=len(threads),
                mininterval=5,
                desc=f"Stacking images of {group_by} dimensions",
                initial=0,
                unit_scale=True,
                colour="cyan",
            ):
                f.result()

            endtime = (time.time() - starttime) / 60
            logger.info(f"Total time taken for execution: {endtime:.4f} minutes")
