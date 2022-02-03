from bfio.bfio import BioReader, BioWriter
from preadator import ProcessManager
from filepattern import FilePattern
import argparse
import logging
from pathlib import Path
import os

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

#TODO: In the future, uncomment this to convert files to the platform file type
# FILE_EXT = FILE_EXT if FILE_EXT is not None else '.ome.tif'
FILE_EXT =os.environ.get('POLUS_EXT','.ome.zarr')

TILE_SIZE = 2 ** 13


def image_converter(inp_image, fileExtension, out_dir):

    FILE_EXT = FILE_EXT if fileExtension is None else fileExtension

    with ProcessManager.process():

        with BioReader(inp_image) as br:

            # Loop through timepoints
            for t in range(br.T):

                # Loop through channels
                for c in range(br.C):

                    extension = "".join(
                        [
                            suffix
                            for suffix in inp_image.suffixes[-2:]
                            if len(suffix) < 6
                        ]
                    )
               
                    out_path = out_dir.joinpath(
                        inp_image.name.replace(extension, FILE_EXT)
                    )
                    if br.C > 1:
                        out_path = out_dir.joinpath(
                            out_path.name.replace(FILE_EXT, f"_c{c}" + FILE_EXT)
                        )
                    if br.T > 1:
                        out_path = out_dir.joinpath(
                            out_path.name.replace(FILE_EXT, f"_t{t}" + FILE_EXT)
                        )

                    with BioWriter(
                        out_path,
                        max_workers=ProcessManager._active_threads,
                        metadata=br.metadata
                    ) as bw:

                        bw.C = 1
                        bw.T = 1
                        bw.channel_names = [br.channel_names[c]]

                        # Loop through z-slices
                        for z in range(br.Z):

                            # Loop across the length of the image
                            for y in range(0, br.Y, TILE_SIZE):
                                y_max = min([br.Y, y + TILE_SIZE])

                                bw.max_workers = ProcessManager._active_threads
                                br.max_workers = ProcessManager._active_threads

                                # Loop across the depth of the image
                                for x in range(0, br.X, TILE_SIZE):
                                    x_max = min([br.X, x + TILE_SIZE])

                                    bw[y:y_max, x:x_max, z : z + 1, 0, 0] = br[
                                        y:y_max, x:x_max, z : z + 1, c, t
                                    ]


def main(
    inpDir: Path,
    filePattern: str,
    fileExtension:str,
    outDir: Path,
) -> None:

    ProcessManager.init_processes("main")

    fp = FilePattern(inpDir, filePattern)

    for files in fp():
        for file in files:
            ProcessManager.submit_process(image_converter, file["file"], fileExtension, outDir)

    ProcessManager.join_processes()


if __name__ == "__main__":

    """Argument parsing"""
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main", description="Convert Bioformats supported format to OME Zarr."
    )

    # Input arguments
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input generic data collection to be processed by this plugin",
        required=True,
    )

    # Input arguments
    parser.add_argument(
        "--filePattern",
        dest="filePattern",
        type=str,
        help="A filepattern defining the images to be converted.",
        required=False,
        default=".*",
    )

    # Input arguments
    parser.add_argument(
        "--fileExtension",
        dest="fileExtension",
        type=str,
        help="Type of data conversion",
        required=False,
        default=FILE_EXT,
    )


    # Output arguments
    parser.add_argument(
        "--outDir", dest="outDir", type=str, help="Output collection", required=True
    )

    # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)
    logger.info("inpDir = {}".format(inpDir))
    filePattern = args.filePattern
    logger.info("filePattern = {}".format(filePattern))
    fileExtension = args.fileExtension
    logger.info("fileExtension = {}".format(fileExtension))
    outDir = Path(args.outDir)
    logger.info("outDir = {}".format(outDir))
    
    main(inpDir=inpDir, filePattern=filePattern, fileExtension=fileExtension, outDir=outDir)