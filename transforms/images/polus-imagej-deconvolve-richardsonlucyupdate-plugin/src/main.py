import typing, os, argparse, logging
import ij_converter
import jpype, imagej, scyjava
import numpy as np
import filepattern
from pathlib import Path
from bfio.bfio import BioReader, BioWriter

"""
This file was automatically generated from an ImageJ plugin generation pipeline.
"""

# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def main(_opName: str, _in1: Path, _regularizationFactor: str, _out: Path,) -> None:

    """ Initialize ImageJ """
    # Bioformats throws a debug message, disable the loci debugger to mute it
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")

    scyjava.when_jvm_starts(disable_loci_logs)

    # This is the version of ImageJ pre-downloaded into the docker container
    logger.info("Starting ImageJ...")

    ij = imagej.init(
        "sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4", headless=True
    )

    logger.info("Loaded ImageJ version: {}".format(ij.getVersion()))

    """ Validate and organize the inputs """
    args = []
    argument_types = []
    arg_len = 0

    # Validate opName
    opName_values = [
        "RichardsonLucyTVUpdate",
        "RichardsonLucyUpdate",
    ]
    assert _opName in opName_values, "opName must be one of {}".format(opName_values)

    # Validate in1
    # in1_types = {
    #     "RichardsonLucyTVUpdate": "RandomAccessibleInterval",
    #     "RichardsonLucyUpdate": "RandomAccessibleInterval",
    # }
    # Validate in1
    in1_types = {
        "RichardsonLucyTVUpdate": "IterableInterval",
        "RichardsonLucyUpdate": "IterableInterval",
    }

    # Check that all inputs are specified
    if _in1 is None and _opName in list(in1_types.keys()):
        raise ValueError("{} must be defined to run {}.".format("in1", _opName))
    elif _in1 != None:
        in1_type = in1_types[_opName]

        # switch to images folder if present
        if _in1.joinpath("images").is_dir():
            _in1 = _in1.joinpath("images").absolute()

        # Check that input path is a directory
        if not _in1.is_dir():
            raise FileNotFoundError(
                "The {} collection directory does not exist".format(_in1)
            )

        # Infer the file pattern of the collection
        pattern_guess = filepattern.infer_pattern(_in1.iterdir())

        # Instantiate the filepatter object
        fp = filepattern.FilePattern(_in1, pattern_guess)

        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        args.append([f[0]["file"] for f in fp() if f[0]["file"].is_file()])
        arg_len = len(args[-1])
    else:
        argument_types.append(None)
        args.append([None])

    # Validate regularizationFactor
    regularizationFactor_types = {
        "RichardsonLucyTVUpdate": "float",
    }

    # Check that all inputs are specified
    if _regularizationFactor is None and _opName in list(
        regularizationFactor_types.keys()
    ):
        raise ValueError(
            "{} must be defined to run {}.".format("regularizationFactor", _opName)
        )
    else:
        regularizationFactor = None

    # This ensures each input collection has the same number of images
    # If one collection is a single image it will be duplicated to match length
    # of the other input collection
    for i in range(len(args)):
        if len(args[i]) == 1:
            args[i] = args[i] * arg_len

    # Define the output data types for each overloading method
    out_types = {
        "RichardsonLucyTVUpdate": "RandomAccessibleInterval",
        "RichardsonLucyUpdate": "RandomAccessibleInterval",
    }
    # Attempt to convert inputs to java types and run the pixel indepent op
    try:
        
        import java.lang.ClassCastException
        import java.lang.IllegalArgumentException
        
        for ind, (in1_path,) in enumerate(zip(*args)):
            if in1_path != None:

                # Load the first plane of image in in1 collection
                logger.info("Processing image: {}".format(in1_path))
                in1_br = BioReader(in1_path)

                # Convert to appropriate numpy array
                in1 = ij_converter.to_java(
                    ij, np.squeeze(in1_br[:, :, 0:1, 0, 0]).astype(float), in1_type
                )
                metadata = in1_br.metadata
                fname = in1_path.name
                dtype = ij.py.dtype(in1)
                # Save the shape for out input
                shape = ij.py.dims(in1)
            if _regularizationFactor is not None:
                regularizationFactor = ij_converter.to_java(
                    ij,
                    _regularizationFactor,
                    regularizationFactor_types[_opName],
                    dtype,
                )

            # Generate the out input variable if required
            out_input = ij_converter.to_java(
                ij, np.empty(shape=shape, dtype=dtype), "IterableInterval"
            )

            logger.info("Running op...")
            if _opName == "RichardsonLucyTVUpdate":
                # out = (
                #     ij.op()
                #     .deconvolve()
                #     .richardsonLucyUpdate(out_input, in1, regularizationFactor)
                # )
                out = (
                    ij.op()
                    .deconvolve()
                    .richardsonLucyUpdate(out_input, in1, regularizationFactor)
                )
            elif _opName == "RichardsonLucyUpdate":
                out = ij.op().deconvolve().richardsonLucyUpdate(out_input, in1)
                # out = ij.op().deconvolve().richardsonLucyUpdate(in1)

            logger.info("Completed op!")
            if in1_path != None:
                in1_br.close()

            # Saving output file to out
            logger.info("Saving...")
            out_array = ij_converter.from_java(ij, out, out_types[_opName])
            bw = BioWriter(_out.joinpath(fname), metadata=metadata)
            bw.Z = 1
            bw.dtype = out_array.dtype
            bw[:] = out_array.astype(bw.dtype)
            bw.close()

    except java.lang.ClassCastException as ex:
        logger.error(ex.stacktrace())
    
    except java.lang.IllegalArgumentException as ex:
        logger.error(ex.stacktrace())

    finally:
        # Exit the program
        logger.info("Shutting down jvm...")
        del ij
        jpype.shutdownJVM()
        logger.info("JVM shutdown complete")


if __name__ == "__main__":

    # Setup Command Line Arguments
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main", description="RichardsonLucyTVUpdate, RichardsonLucyUpdate"
    )

    # Add command-line argument for each of the input arguments
    parser.add_argument(
        "--opName", dest="opName", type=str, help="Operation to perform", required=False
    )
    parser.add_argument("--in1", dest="in1", type=str, help="in1", required=False)
    parser.add_argument(
        "--regularizationFactor",
        dest="regularizationFactor",
        type=str,
        help="regularizationFactor",
        required=False,
    )

    # Add command-line argument for each of the output arguments
    parser.add_argument("--out", dest="out", type=str, help="out", required=True)

    """ Parse the arguments """
    args = parser.parse_args()

    # Input Args
    _opName = args.opName
    logger.info("opName = {}".format(_opName))

    _in1 = Path(args.in1)
    logger.info("in1 = {}".format(_in1))

    _regularizationFactor = args.regularizationFactor
    logger.info("regularizationFactor = {}".format(_regularizationFactor))

    # Output Args
    _out = Path(args.out)
    logger.info("out = {}".format(_out))

    main(
        _opName=_opName,
        _in1=_in1,
        _regularizationFactor=_regularizationFactor,
        _out=_out,
    )
