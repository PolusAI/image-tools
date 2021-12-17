from pathlib import Path
import sys

src_dir = Path(__file__).parents[1].joinpath("{{cookiecutter.project_slug}}/src")
print(src_dir)
sys.path.append(str(src_dir))

import ij_converter

if __name__ == "__main__":
    import imagej
    import scyjava
    import traceback
    import sys
    import imglyb
    import jpype
    import numpy as np
    from bfio import BioReader, BioWriter
    from pathlib import Path
    # Bioformats throws a debug message, disable the loci debugger to mute it
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")

    scyjava.when_jvm_starts(disable_loci_logs)

    print("Starting JVM...")

    # This is the version of ImageJ pre-downloaded into the docker container
    ij = imagej.init(
        "sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4", headless=True
    )

    # ArrayImgs = scyjava.jimport('net.imglib2.img.array.ArrayImgs')
    # UnsafeUtil = scyjava.jimport('net.imglib2.img.basictypelongaccess.unsafe.UnsafeUtil')
    # Arrays = scyjava.jimport('java.util.Arrays')
    # OwningFloatUnsafe = scyjava.jimport('net.imglib2.img.basictypelongaccess.unsafe.owning.OwningFloatUnsafe')
    # Fraction = scyjava.jimport('net.imglib2.util.Fraction')
    # LongStream = scyjava.jimport('java.util.stream.LongStream')

    NUMPY_TYPES = {
        "uint8": (np.uint8, imglyb.types.UnsignedByteType),
        "byte": (np.int8, imglyb.types.ByteType),
        "uint16": (np.uint16, imglyb.types.UnsignedShortType),
        "short": (np.int16, imglyb.types.ShortType),
        "uint32": (np.uint32, imglyb.types.UnsignedIntType),
        "int": (np.int32, imglyb.types.IntType),
        "float": (np.float32, imglyb.types.DoubleType),
        "double": (np.float64, imglyb.types.UnsignedLongType),
        "long": (np.int64, imglyb.types.LongType),
    }

    TEST_CASES = ["double", "float", "long", "int", "short", "char", "byte", "boolean"]

    def tester(t, ij):
        try:
            print("Testing {} data type...".format(t))
            shape = (2048, 2048)
            print("Creating Array...")
            array = np.random.randint(0, 255, size=shape, dtype=np.uint16)
            print("Converting Array...")
            array = NUMPY_TYPES[t][0](array)
            dtype0 = ij.py.dtype(array)
            print("The initial data type is {}".format(dtype0))
            temp_path = Path(__file__).with_name("data-convert-temp")
            print("Writing image array to file...")
            with BioWriter(temp_path) as writer:
                writer.X = shape[0]
                writer.Y = shape[1]
                writer.dtype = array.dtype
                writer[:] = array[:]
            print("Reading image from file...")
            arr = BioReader(temp_path)
            print("Getting data type after reading image...")
            dtype1 = ij.py.dtype(arr[:, :, 0:1, 0, 0])
            print("Data type after reading image is {}".format(dtype1))
            # print('Trying to convert to PlanarImg')
            # planarimg = ij.planar(arr)
            if dtype0 != dtype1:
                print("Manully forcing data type back to {}".format(dtype0))
                arr = NUMPY_TYPES[t][0](arr[:, :, 0:1, 0, 0])
                print("Converting to Java object...")
                arr = ij_converter.to_java(ij, np.squeeze(arr), "ArrayImg")
                print("Getting data type after manually forcing...")
                dtype2 = ij.py.dtype(arr)
                print("Data type after manual forcing is {}".format(dtype2))
                val_dtype = dtype2
            else:
                arr = ij_converter.to_java(
                    ij, np.squeeze(arr[:, :, 0:1, 0, 0]), "ArrayImg"
                )
                val_dtype = dtype1

            value = 5
            print(
                "Converting input (value) to Java primitive type {}...".format(
                    val_dtype
                )
            )
            val = ij_converter.to_java(ij, value, t, val_dtype)
            print("Calling ImageJ op...")
            out = ij.op().math().add(arr, val)
            print("The op was SUCCESSFUL with data type {}".format(t))

        except:
            print("Testing data type {} was NOT SUCCESSFUL".format(t))
            print(traceback.format_exc())

        finally:
            print("Shutting down JVM...\n\n")
            del ij
            jpype.shutdownJVM()

    tester(TEST_CASES[1], ij)

    """Example of how to cast multidimensional arrays to java"""
    z = np.zeros((5, 10, 20))
    ja = jpype.JArray.of(z)
    print(type(ja), "\n\n")
