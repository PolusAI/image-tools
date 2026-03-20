"""Utilities for type conversion between Python and Java."""


import enum
import typing

import numpy
from imagej import convert
from scyjava import jimport


class IjType(str, enum.Enum):
    """Enum for ImageJ data types."""

    UnsignedByte = "uint8"
    SignedByte = "int8"
    UnsignedShort = "uint16"
    SignedShort = "int16"
    UnsignedInt = "uint32"
    SignedInt = "int32"
    Float = "float32"
    Double = "float64"

    @staticmethod
    def from_dtype(dtype: numpy.dtype) -> "IjType":
        """Get the ImageJ type from the numpy dtype."""
        if dtype == numpy.uint8:
            val = IjType.UnsignedByte
        elif dtype == numpy.int8:
            val = IjType.SignedByte
        elif dtype == numpy.uint16:
            val = IjType.UnsignedShort
        elif dtype == numpy.int16:
            val = IjType.SignedShort
        elif dtype == numpy.uint32:
            val = IjType.UnsignedInt
        elif dtype == numpy.int32:
            val = IjType.SignedInt
        elif dtype == numpy.float32:
            val = IjType.Float
        elif dtype == numpy.float64:
            val = IjType.Double
        else:
            msg = f"Unsupported numpy dtype: {dtype}"
            raise ValueError(msg)
        return val

    def cast_primitive(self, value: typing.Any) -> typing.Any:
        """Cast the value to the corresponding Java type."""
        if self == IjType.UnsignedByte:
            tp = jimport("net.imglib2.type.numeric.integer.UnsignedByteType")
        elif self == IjType.SignedByte:
            tp = jimport("net.imglib2.type.numeric.integer.ByteType")
        elif self == IjType.UnsignedShort:
            tp = jimport("net.imglib2.type.numeric.integer.UnsignedShortType")
        elif self == IjType.SignedShort:
            tp = jimport("net.imglib2.type.numeric.integer.ShortType")
        elif self == IjType.UnsignedInt:
            tp = jimport("net.imglib2.type.numeric.integer.UnsignedIntType")
        elif self == IjType.SignedInt:
            tp = jimport("net.imglib2.type.numeric.integer.IntType")
        elif self == IjType.Float:
            tp = jimport("net.imglib2.type.numeric.real.FloatType")
        elif self == IjType.Double:
            tp = jimport("net.imglib2.type.numeric.real.DoubleType")
        else:
            msg = f"Unsupported ImageJ type: {self}"
            raise ValueError(msg)

        return tp(value)

    def cast_image_to_ij(self, ij: typing.Any, value: numpy.ndarray) -> typing.Any:
        """Cast the value to the corresponding Java type."""
        if ij is None:
            msg = "No imagej instance found."
            raise ValueError(msg)

        return convert.ndarray_to_img(ij, value)

    def cast_ij_to_image(self, ij: typing.Any, value: typing.Any) -> numpy.ndarray:
        """Cast the value to the corresponding Java type."""
        if ij is None:
            msg = "No imagej instance found."
            raise ValueError(msg)

        img = convert.java_to_ndarray(ij, value)
        if self == IjType.UnsignedByte:
            img = img.astype(numpy.uint8)
        elif self == IjType.SignedByte:
            img = img.astype(numpy.int8)
        elif self == IjType.UnsignedShort:
            img = img.astype(numpy.uint16)
        elif self == IjType.SignedShort:
            img = img.astype(numpy.int16)
        elif self == IjType.UnsignedInt:
            img = img.astype(numpy.uint32)
        elif self == IjType.SignedInt:
            img = img.astype(numpy.int32)
        elif self == IjType.Float:
            img = img.astype(numpy.float32)
        elif self == IjType.Double:
            img = img.astype(numpy.float64)
        else:
            msg = f"Unsupported ImageJ type: {self}"
            raise ValueError(msg)

        return img
