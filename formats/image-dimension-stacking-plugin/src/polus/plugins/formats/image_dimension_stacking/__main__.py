"""Ome micojson package."""
import logging
import shutil
import time
import warnings
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from os import environ
from pathlib import Path
from typing import List
import numpy as np

import filepattern as fp

# import polus.plugins.formats.image_dimension_stacking as st
import typer
from tqdm import tqdm
import pprint
from bfio import BioReader, BioWriter

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.formats.axis_stacking")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)


app = typer.Typer(help="Stack multi dimensional image into single image.")


# def generate_preview(
#     out_dir: Path,
# ) -> None:
#     """Generate preview of the plugin outputs."""
#     shutil.copy(
#         Path(__file__).parents[5].joinpath("segmentations.json"),
#         out_dir,
#     )


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to input directory containing binary images.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filename pattern used to separate data.",
    ),
    channel_order: str = typer.Option(
        ...,
        "--channelOrder",
        "-c",
        help="Desired polygon type.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Generate preview of expected outputs.",
    ),
) -> None:
    """Convert binary segmentations to micojson."""
    logger.info(f"--inpDir: {inp_dir}")
    logger.info(f"--filePattern: {file_pattern}")
    logger.info(f"--channelOrder: {channel_order}")
    logger.info(f"--outDir: {out_dir}")
    starttime = time.time()

    # if not inp_dir.exists():
    #     msg = "inpDir does not exist"
    #     raise ValueError(msg, inp_dir)

    # if not out_dir.exists():
    #     msg = "outDir does not exist"
    #     raise ValueError(msg, out_dir)

    # def save_ometif(image: np.ndarray, out_file: Path, c:int) -> None:
    #     """Write ome tif image."""
    #     with BioWriter(file_path=out_file) as bw:
    #         bw.X = image.shape[1]
    #         bw.Y = image.shape[0]
    #         bw.Z = 1
    #         bw.C = c
    #         bw.T = 1
    #         bw.dtype = image.dtype
    #         bw[:] = image

    # channel_order = [int(c) for c in channel_order.split(',')]

    # files = fp.FilePattern(inp_dir, file_pattern)

    # if not len(files) > 0:
    #     msg = "No image files are detected. Please check filepattern again!"
    #     raise ValueError(msg)

    # f_group = files(group_by='c')

    # paths = []
    # for c in channel_order:
    #     for file in f_group:
    #         _, fi = file
    #         f1, f2 = zip(*fi)
    #         if f1[0]['c'] == c:
    #             paths.append(f2[0][0])
    #             break
    # tile_size = 1024
    # for c, p in zip(channel_order, paths):
    #     # br = BioReader(p)
    #     # image = br.read()
    #     file_name = files.output_name()
    #     logger.info('Writing: {}'.format(file_name))

    #     with BioReader(p,max_workers=cpu_count()) as br:

    #         with BioWriter(file_name,
    #                     backend='python',
    #                     metadata=br.metadata,
    #                     max_workers = cpu_count()) as bw:
    #             bw.C = c
    #             bw.X = br.X
    #             bw.Y = br.Y
    #             bw.Z = br.Z
    #             bw.T = br.T
    #             bw.dtype = br.dtype
    #             bw[:] = br[:, :, 1, c, 1]

    # # Loop across the length of the image
    # for y in range(0,br.Y,tile_size):
    #     y_max = min([br.Y,y+tile_size])

    #     # Loop across the depth of the image
    #     for x in range(0,br.X,tile_size):
    #         x_max = min([br.X,x+tile_size])

    #         bw[y:y_max,x:x_max,0,c:c+1,0] = br[y:y_max,x:x_max,0,c:c+1,0]

    # save_ometif(image=image, out_file=file_name, c=c)
    # bw = BioWriter(str(Path(out_dir).joinpath(file_name)),
    #                    metadata=br.metadata)

    # with ProcessPoolExecutor(max_workers=sm.NUM_THREADS) as executor:
    #     for _, f in enumerate(tqdm(files())):
    #         model = sm.OmeMicrojsonModel(
    #             out_dir=out_dir,
    #             file_path=str(f[1][0]),
    #             polygon_type=polygon_type,
    #         )
    #         executor.submit(model.write_single_json())

    # if preview:
    #     generate_preview(out_dir)
    #     logger.info(f"generating preview data in {out_dir}")

    # endtime = (time.time() - starttime) / 60
    # logger.info(f"Total time taken for execution: {endtime:.4f} minutes")
    # logger.info(f"Total time taken for a single file: {endtime/len(files):.4f} minutes")


if __name__ == "__main__":
    app()
