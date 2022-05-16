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


def main(
    _opName: str,
    _in1: Path,
    _in2: Path,
    _fftBuffer: Path,
    _fftKernel: Path,
    _out: Path,
) -> None:

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
        "RichardsonLucyCorrection",
    ]
    assert _opName in opName_values, "opName must be one of {}".format(opName_values)

    # Validate in1
    in1_types = {
        "RichardsonLucyCorrection": "RandomAccessibleInterval",
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

    # Validate in2
    in2_types = {
        "RichardsonLucyCorrection": "RandomAccessibleInterval",
    }

    # Check that all inputs are specified
    if _in2 is None and _opName in list(in2_types.keys()):
        raise ValueError("{} must be defined to run {}.".format("in2", _opName))
    elif _in2 != None:
        in2_type = in2_types[_opName]

        # switch to images folder if present
        if _in2.joinpath("images").is_dir():
            _in2 = _in2.joinpath("images").absolute()

        # Check that input path is a directory
        if not _in2.is_dir():
            raise FileNotFoundError(
                "The {} collection directory does not exist".format(_in2)
            )

        # Infer the file pattern of the collection
        pattern_guess = filepattern.infer_pattern(_in2.iterdir())

        # Instantiate the filepatter object
        fp = filepattern.FilePattern(_in2, pattern_guess)

        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        args.append([f[0]["file"] for f in fp() if f[0]["file"].is_file()])
        arg_len = len(args[-1])
    else:
        argument_types.append(None)
        args.append([None])

    # Validate fftBuffer
    fftBuffer_types = {
        "RichardsonLucyCorrection": "RandomAccessibleInterval",
    }

    # Check that all inputs are specified
    if _fftBuffer is None and _opName in list(fftBuffer_types.keys()):
        raise ValueError("{} must be defined to run {}.".format("fftBuffer", _opName))
    elif _fftBuffer != None:
        fftBuffer_type = fftBuffer_types[_opName]

        # switch to images folder if present
        if _fftBuffer.joinpath("images").is_dir():
            _fftBuffer = _fftBuffer.joinpath("images").absolute()

        # Check that input path is a directory
        if not _fftBuffer.is_dir():
            raise FileNotFoundError(
                "The {} collection directory does not exist".format(_fftBuffer)
            )

        # Infer the file pattern of the collection
        pattern_guess = filepattern.infer_pattern(_fftBuffer.iterdir())

        # Instantiate the filepatter object
        fp = filepattern.FilePattern(_fftBuffer, pattern_guess)

        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        args.append([f[0]["file"] for f in fp() if f[0]["file"].is_file()])
        arg_len = len(args[-1])
    else:
        argument_types.append(None)
        args.append([None])

    # Validate fftKernel
    fftKernel_types = {
        "RichardsonLucyCorrection": "RandomAccessibleInterval",
    }

    # Check that all inputs are specified
    if _fftKernel is None and _opName in list(fftKernel_types.keys()):
        raise ValueError("{} must be defined to run {}.".format("fftKernel", _opName))
    elif _fftKernel != None:
        fftKernel_type = fftKernel_types[_opName]

        # switch to images folder if present
        if _fftKernel.joinpath("images").is_dir():
            _fftKernel = _fftKernel.joinpath("images").absolute()

        # Check that input path is a directory
        if not _fftKernel.is_dir():
            raise FileNotFoundError(
                "The {} collection directory does not exist".format(_fftKernel)
            )

        # Infer the file pattern of the collection
        pattern_guess = filepattern.infer_pattern(_fftKernel.iterdir())

        # Instantiate the filepatter object
        fp = filepattern.FilePattern(_fftKernel, pattern_guess)

        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        args.append([f[0]["file"] for f in fp() if f[0]["file"].is_file()])
        arg_len = len(args[-1])
    else:
        argument_types.append(None)
        args.append([None])

    # This ensures each input collection has the same number of images
    # If one collection is a single image it will be duplicated to match length
    # of the other input collection
    for i in range(len(args)):
        if len(args[i]) == 1:
            args[i] = args[i] * arg_len

    # Define the output data types for each overloading method
    out_types = {
        "RichardsonLucyCorrection": "RandomAccessibleInterval",
    }
    # Attempt to convert inputs to java types and run the pixel indepent op
    try:
        import java.lang.IllegalArgumentException
        for ind, (in1_path, in2_path, fftBuffer_path, fftKernel_path,) in enumerate(
            zip(*args)
        ):
            if in1_path != None:

                # Load the first plane of image in in1 collection
                logger.info("Processing image: {}".format(in1_path))
                in1_br = BioReader(in1_path)

                # Convert to appropriate numpy array
                in1 = ij_converter.to_java(
                    ij, np.squeeze(in1_br[:, :, 0:1, 0, 0]), in1_type
                )
                metadata = in1_br.metadata
                fname = in1_path.name
                dtype = ij.py.dtype(in1)
                # Save the shape for out input
                shape = ij.py.dims(in1)
            if in2_path != None:

                # Load the first plane of image in in2 collection
                logger.info("Processing image: {}".format(in2_path))
                in2_br = BioReader(in2_path)

                # Convert to appropriate numpy array
                in2 = ij_converter.to_java(
                    ij, np.squeeze(in2_br[:, :, 0:1, 0, 0]), in2_type
                )
            if fftBuffer_path != None:

                # Load the first plane of image in fftBuffer collection
                logger.info("Processing image: {}".format(fftBuffer_path))
                fftBuffer_br = BioReader(fftBuffer_path)

                # Convert to appropriate numpy array
                fftBuffer = ij_converter.to_java(
                    ij, np.squeeze(fftBuffer_br[:, :, 0:1, 0, 0]), fftBuffer_type
                )
            if fftKernel_path != None:

                # Load the first plane of image in fftKernel collection
                logger.info("Processing image: {}".format(fftKernel_path))
                fftKernel_br = BioReader(fftKernel_path)

                # Convert to appropriate numpy array
                fftKernel = ij_converter.to_java(
                    ij, np.squeeze(fftKernel_br[:, :, 0:1, 0, 0]), fftKernel_type
                )

            # Generate the out input variable if required
            out_input = ij_converter.to_java(
                ij, np.zeros(shape=shape, dtype=dtype), "RandomAccessibleInterval"
            )
            
            logger.info("Running op...")
            if _opName == "RichardsonLucyCorrection":
                # out = ij.op().deconvolve().richardsonLucyCorrection(
                #     out_input, 
                #     in1, 
                #     in2, 
                #     fftBuffer, 
                #     fftKernel
                #     )
                out = ij.op().deconvolve().richardsonLucyCorrection(
                    out_input, 
                    in1, 
                    in2, 
                    fftBuffer, 
                    fftKernel
                    )

            logger.info("Completed op!")
            if in1_path != None:
                in1_br.close()
            if in2_path != None:
                in2_br.close()
            if fftBuffer_path != None:
                fftBuffer_br.close()
            if fftKernel_path != None:
                fftKernel_br.close()

            # Saving output file to out
            logger.info("Saving...")
            out_array = ij_converter.from_java(ij, out, out_types[_opName])
            bw = BioWriter(_out.joinpath(fname), metadata=metadata)
            bw.Z = 1
            bw.dtype = out_array.dtype
            bw[:] = out_array.astype(bw.dtype)
            bw.close()

    except java.lang.IllegalArgumentException as ex:
        logger.error("There was an error, shutting down jvm...")
        logger.error(ex.stacktrace())
        raise

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
        prog="main", description="RichardsonLucyCorrection"
    )

    # Add command-line argument for each of the input arguments
    parser.add_argument(
        "--opName", dest="opName", type=str, help="Operation to perform", required=False
    )
    parser.add_argument("--in1", dest="in1", type=str, help="in1", required=False)
    parser.add_argument("--in2", dest="in2", type=str, help="in2", required=False)
    parser.add_argument(
        "--fftBuffer", dest="fftBuffer", type=str, help="fftBuffer", required=False
    )
    parser.add_argument(
        "--fftKernel", dest="fftKernel", type=str, help="fftKernel", required=False
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

    _in2 = Path(args.in2)
    logger.info("in2 = {}".format(_in2))

    _fftBuffer = Path(args.fftBuffer)
    logger.info("fftBuffer = {}".format(_fftBuffer))

    _fftKernel = Path(args.fftKernel)
    logger.info("fftKernel = {}".format(_fftKernel))

    # Output Args
    _out = Path(args.out)
    logger.info("out = {}".format(_out))

    main(
        _opName=_opName,
        _in1=_in1,
        _in2=_in2,
        _fftBuffer=_fftBuffer,
        _fftKernel=_fftKernel,
        _out=_out,
    )
