"""
A conversion utility built to convert abstract to primitive
"""

import imagej
import logging
import imglyb
import jpype
import numpy as np
import scyjava

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("ij_converter")
logger.setLevel(logging.INFO)

## fill in types to convert
ABSTRACT_ITERABLES = [
    "IterableInterval",
    "Iterable",
]

IMG_ARRAYS = ["ArrayImg"]

ABSTRACT_SCALARS = [
    "RealType",
]

SCALARS = [
    "double",
    "float",
    "long",  # long type (int64) not supported by bfio
    "int",
    "short",
    "char",
    "byte",
    "boolean",
]

## recognize array objects as scalar objects + '[]'
ARRAYS = [s + "[]" for s in SCALARS]


def _java_setup():
    global IMGLYB_PRIMITIVES, PRIMITIVES, PRIMITIVE_ARRAYS
    IMGLYB_PRIMITIVES = {
        "float32": imglyb.types.FloatType,
        "float64": imglyb.types.DoubleType,
        "int8": imglyb.types.ByteType,
        "int16": imglyb.types.ShortType,
        "int32": imglyb.types.IntType,
        "int64": imglyb.types.LongType,
        "uint8": imglyb.types.UnsignedByteType,
        "uint16": imglyb.types.UnsignedShortType,
        "uint32": imglyb.types.UnsignedIntType,
        "uint64": imglyb.types.UnsignedLongType,
    }
    # PRIMITIVES = {
    #     'double'     : jpype.JDouble,
    #     'float'      : jpype.JFloat,
    #     'long'       : jpype.JLong,
    #     'int'        : jpype.JInt,
    #     'short'      : jpype.JShort,
    #     'char'       : jpype.JChar,
    #     'byte'       : jpype.JByte,
    #     'boolean'    : jpype.JBoolean
    # }
    PRIMITIVES = {
        "double": jpype.JDouble,
        "float": jpype.JFloat,
        "long": jpype.JLong,
        "int": jpype.JInt,
        "short": jpype.JShort,
        "char": jpype.JChar,
        "byte": jpype.JByte,
        "boolean": jpype.JBoolean,
    }
    PRIMITIVE_ARRAYS = {
        "double[]": jpype.JDouble[:],
        "float[]": jpype.JFloat[:],
        "long[]": jpype.JLong[:],
        "int[]": jpype.JInt[:],
        "short[]": jpype.JShort[:],
        "char[]": jpype.JChar[:],
        "byte[]": jpype.JByte[:],
        "boolean[]": jpype.JBoolean[:],
    }


scyjava.when_jvm_starts(_java_setup)

JAVA_CONVERT = {}

JAVA_CONVERT.update(
    {
        t: lambda s, t, st: IMGLYB_PRIMITIVES[str(st)](st.type(s))
        for t in ABSTRACT_SCALARS
    }
)
JAVA_CONVERT.update({t: lambda s, t, st: PRIMITIVES[t](float(s)) for t in SCALARS})
JAVA_CONVERT.update(
    {
        t: lambda s, t, st: PRIMITIVE_ARRAYS[t]([float(si) for si in s.split(",")])
        for t in ARRAYS
    }
)
# JAVA_CONVERT.update({
#     t: lambda s,t,st: IMGLYB_PRIMITIVES[str(st)](s) for t in SCALARS
# })

JAVA_CONVERT.update(
    {
        t: lambda s, ij: imglyb.util.Views.iterable(ij.py.to_java(s))
        for t in ABSTRACT_ITERABLES
    }
)
JAVA_CONVERT.update({t: lambda s, ij: imglyb.util._to_imglib(s) for t in IMG_ARRAYS})


def to_java(ij, np_array, java_type, java_dtype=None):

    if ij == None:
        raise ValueError("No imagej instance found.")

    if isinstance(np_array, type(None)):
        return None

    if java_type in JAVA_CONVERT.keys():
        if str(java_dtype) != "None":
            out_array = JAVA_CONVERT[java_type](np_array, java_type, java_dtype)
        else:
            out_array = JAVA_CONVERT[java_type](np_array, ij)
    else:
        logger.warning(
            "Did not recognize type, {}, will pass default.".format(java_type)
        )
        out_array = ij.py.to_java(np_array)

    return out_array


def from_java(ij, java_array, java_type):

    if ij == None:
        raise ValueError("No imagej instance found.")

    if ij.py.dtype(java_array) == bool:
        java_array = ij.op().convert().uint8(java_array)

    return ij.py.from_java(java_array)
