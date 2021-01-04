"""
A conversion utility built to convert abstract to primitive
This works for Threshold Apply and Gaussian Filter functions
Note change from jnius to jpype for handling conversions in ij_converter.py

'''

import logging
import imglyb
import jpype
import scyjava
import numpy as np

##connect type map from populate.py 
from populate.py import IMAGEJ_WIPP_TYPE

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("ij_converter")
logger.setLevel(logging.INFO)

ij = None

## fill in types to convert
ABSTRACT_ITERABLES = [
    'IterableInterval',
    'Iterable'
]

ABSTRACT_SCALARS = [
    'RealType'
]

SCALARS = [
    'double'
]

INT_PRIMITIVES = [
    "int",
    "short"
]

CHAR_PRIMITIVES = [
    "char"
]

BYTE_PRIMITIVES = [
    "byte"
]

BOOL_PRIMITIVES = [
    "boolean"
]

# Recognize array objects as primitive objects + '[]'
FLOAT_ARRAYS = [s + "[]" for s in FLOAT_PRIMITIVES]
INT_ARRAYS = [s + "[]" for s in INT_PRIMITIVES]
CHAR_ARRAYS = [s + "[]" for s in CHAR_PRIMITIVES]
BYTE_ARRAYS = [s + "[]" for s in BYTE_PRIMITIVES]
BOOL_ARRAYS = [s + "[]" for s in BOOL_PRIMITIVES]


def _java_setup():
    global IMGLYB_PRIMITIVES, PRIMITIVES, PRIMITIVE_FLOAT_ARRAYS, PRIMITIVE_INT_ARRAYS
    global PRIMITIVE_CHAR_ARRAYS, PRIMITIVE_BYTE_ARRAYS, PRIMITIVE_BOOL_ARRAYS
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
    PRIMITIVE_FLOAT_ARRAYS = {
        "double[]": jpype.JDouble[:],
        "float[]": jpype.JFloat[:]
    }
scyjava.when_jvm_starts(_java_setup)

# Define empty dictionary to store the data type conversion functions
JAVA_CONVERT = {}
JAVA_CONVERT.update({
    t:lambda s,t,st: IMGLYB_PRIMITIVES[str(st)](st.type(s)) for t in ABSTRACT_SCALARS
})
JAVA_CONVERT.update({
    t:lambda s,t,st: PRIMITIVES[t](float(s)) for t in SCALARS
})
JAVA_CONVERT.update({
    t:lambda s,t,st: PRIMITIVE_ARRAYS[t]([float(si) for si in s.split(',')]) for t in ARRAYS
})
JAVA_CONVERT.update({
    t:lambda s: imglyb.util.Views.iterable(ij.py.to_java(s)) for t in ABSTRACT_ITERABLES
})

def to_java(np_array,java_type,java_dtype=None):

    if ij == None:
        raise ValueError("No imagej instance found.")

    if isinstance(np_array, type(None)):
        return jpype.JObject(None, type)
    
    # TODO: Define how null objects should be converted from python to java 
    # if java_type == "null":
    #     return jpype.JObject(None, type)

    if java_type in JAVA_CONVERT.keys():
        if java_dtype != None:
            out_array = JAVA_CONVERT[java_type](np_array,java_type,java_dtype)
        else:
            out_array = JAVA_CONVERT[java_type](np_array)
    else:
        logger.warning(
            "Did not recognize type, {}, will pass default.".format(imagej_type)
        )
        # Converts to RandomAccesibleInterval if imagej type not recognized
        out_array = ij.py.to_java(np_array)

    return out_array

def from_java(java_array,java_type):

    if ij == None:
        raise ValueError("No imagej instance found.")

    if ij.py.dtype(java_array) == bool:
        java_array = ij.op().convert().uint8(java_array)

    return ij.py.from_java(java_array) 
