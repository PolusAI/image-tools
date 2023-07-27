"""Utility functions for precompute slide plugin."""

import abc
import concurrent.futures
import copy
import enum
import json
import logging
import os
import pathlib
import typing

import bfio
import imageio
import numpy as np
import zarr
from bfio.OmeXml import OMEXML
from numcodecs import Blosc
from preadator import ProcessManager

logging.getLogger("bfio").setLevel(logging.CRITICAL)

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))

# Chunk Scale
CHUNK_SIZE = 1024

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {
    "m": 10**9,  # meters
    "cm": 10**7,  # centimeters
    "mm": 10**6,  # millimeters
    "µm": 10**3,  # micrometers
    "nm": 1,  # nanometers, default
    "Å": 10**-1,  # angstroms
}


class ImageType(str, enum.Enum):
    """Image type for precomputed format."""

    Image = "image"
    Segmentation = "segmentation"

    def scale(self) -> typing.Callable[[np.ndarray], np.ndarray]:
        """Get the scaling function for the image type."""
        if self == ImageType.Image:
            return _avg2

        return _mode2

    @classmethod
    def variants(cls) -> list["ImageType"]:
        """Get a list of all image types."""
        return [ImageType.Image, ImageType.Segmentation]


class PyramidType(str, enum.Enum):
    """Pyramid type for precomputed format."""

    Neuroglancer = "Neuroglancer"
    DeepZoom = "DeepZoom"
    Zarr = "Zarr"

    def create(self, *args, **kwargs) -> "PyramidWriter":  # noqa: ANN002, ANN003
        """Create a new PyramidWriter object based on the pyramid type."""
        if self == PyramidType.Neuroglancer:
            return NeuroglancerWriter(*args, **kwargs)  # type: ignore

        if self == PyramidType.DeepZoom:
            return DeepZoomWriter(*args, **kwargs)

        return ZarrWriter(*args, **kwargs)  # type: ignore

    @classmethod
    def variants(cls) -> list["PyramidType"]:
        """Get a list of all pyramid types."""
        return [PyramidType.DeepZoom, PyramidType.Neuroglancer, PyramidType.Zarr]


def _mode2(image: np.ndarray) -> np.ndarray:
    """Find mode of pixels in optical field 2x2 and stride 2.

    This method approximates the mode by finding the largest number that occurs
    at least twice in a 2x2 grid of pixels, then sets that value to the output
    pixel.

    Args:
        image: numpy array with only two dimensions (m,n)

    Returns:
        numpy array with only two dimensions (round(m/2),round(n/2))
    """
    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2

    # Initialize the mode output image (Half the size)
    mode_img = np.zeros(
        np.ceil([d / 2 for d in image.shape]).astype(int),
        dtype=image.dtype,
    )

    # Default the output to the upper left pixel value
    mode_img[0 : y_max // 2, 0 : x_max // 2] = image[0:-1:2, 0:-1:2]

    # Handle images with odd-valued image dimensions
    if y_max != image.shape[0]:
        mode_img[-1, : x_max // 2] = image[-1, 0 : x_max - 1 : 2]
    if x_max != image.shape[1]:
        mode_img[: y_max // 2, -1] = image[0 : y_max - 1 : 2, -1]
    if y_max != image.shape[0] and x_max != image.shape[1]:
        mode_img[-1, -1] = image[-1, -1]

    # Garnering the four different pixels that we would find the modes of
    # Finding the mode of:
    # etc
    vals00 = image[0:-1:2, 0:-1:2]
    vals01 = image[0:-1:2, 1::2]
    vals10 = image[1::2, 0:-1:2]
    vals11 = image[1::2, 1::2]

    # Finding where pixels adjacent to the top left pixel are not identical
    index = (vals00 != vals01) | (vals00 != vals10)

    # Initialize indexes where the two pixels are not the same
    valueslist = [vals00[index], vals01[index], vals10[index], vals11[index]]

    # Do a deeper mode search for non-matching pixels
    temp_mode = mode_img[: y_max // 2, : x_max // 2]
    for i in range(3):
        rvals = valueslist[i]
        for j in range(i + 1, 4):
            cvals = valueslist[j]
            ind = np.logical_and(cvals == rvals, rvals > temp_mode[index])
            temp_mode[index][ind] = rvals[ind]

    mode_img[: y_max // 2, : x_max // 2] = temp_mode

    return mode_img


def _avg2(image: np.ndarray) -> np.ndarray:
    """Average pixels together with optical field 2x2 and stride 2.

    Args:
        image: numpy array with only two dimensions (m,n)

    Returns:
        numpy array with only two dimensions (round(m/2),round(n/2))
    """
    # Since we are adding pixel values, we need to update the pixel type
    # This helps to avoid integer overflow
    if image.dtype == np.uint8:
        dtype = np.uint16
    elif image.dtype == np.uint16:
        dtype = np.uint32
    elif image.dtype == np.uint32:
        dtype = np.uint64
    elif image.dtype == np.int8:
        dtype = np.int16
    elif image.dtype == np.int16:
        dtype = np.int32
    elif image.dtype == np.int32:
        dtype = np.int64
    else:
        dtype = image.dtype

    odtype = image.dtype
    image = image.astype(dtype)

    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2

    # Calculate the mean
    avg_img = np.zeros(np.ceil([d / 2 for d in image.shape]).astype(int), dtype=dtype)
    avg_img[0 : y_max // 2, 0 : x_max // 2] = (
        image[0 : y_max - 1 : 2, 0 : x_max - 1 : 2]
        + image[1:y_max:2, 0 : x_max - 1 : 2]
        + image[0 : y_max - 1 : 2, 1:x_max:2]
        + image[1:y_max:2, 1:x_max:2]
    ) // 4

    # Fill in the final row if the image height is odd-valued
    if y_max != image.shape[0]:
        avg_img[-1, : x_max // 2] = (
            image[-1, 0 : x_max - 1 : 2] + image[-1, 1:x_max:2]
        ) // 2
    # Fill in the final column if the image width is odd-valued
    if x_max != image.shape[1]:
        avg_img[: y_max // 2, -1] = (
            image[0 : y_max - 1 : 2, -1] + image[1:y_max:2, -1]
        ) // 2
    # Fill in the lower right pixel if both image width and height are odd
    if y_max != image.shape[0] and x_max != image.shape[1]:
        avg_img[-1, -1] = image[-1, -1]

    return avg_img.astype(odtype)


# Modified and condensed from FileAccessor class in neuroglancer-scripts
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/file_accessor.py
class PyramidWriter:
    """Pyramid file writing base class.

    This class should not be called directly. It should be inherited by a pyramid
    writing class type.

    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    chunk_pattern = ""

    def __init__(  # noqa: PLR0913
        self,
        base_dir: pathlib.Path,
        image_path: pathlib.Path,
        image_depth: int = 0,
        output_depth: int = 0,
        max_output_depth: int = 0,
        image_type: ImageType = ImageType.Image,
    ) -> None:
        """Initialize the pyramid writer class.

        Args:
            base_dir: The base directory where the pyramid will be written.
            image_path: The path to the image file.
            image_depth: The depth of the image file.
            output_depth: The depth of the output pyramid.
            max_output_depth: The maximum depth of the output pyramid.
            image_type: The type of image to be written.
        """
        self.image_path = image_path
        self.base_path = base_dir
        self.image_depth = image_depth
        self.output_depth = output_depth
        self.max_output_depth = max_output_depth
        self.image_type = image_type.value
        self.scale = image_type.scale()

        self.info = bfio_metadata_to_slide_info(
            self.image_path,
            self.max_output_depth,
            self.image_type,
        )

        self.dtype = self.info["data_type"]

        self.encoder = self._encoder()

    @abc.abstractmethod
    def _encoder(self):  # noqa: ANN202
        pass

    @abc.abstractmethod
    def _write_chunk(
        self,
        key: str,
        chunk_coords: tuple[int, ...],
        buf: np.array,
    ) -> None:
        pass

    @abc.abstractmethod
    def write_info(self):  # noqa: D102, ANN201
        pass

    @abc.abstractmethod
    def write_segment_info(self):  # noqa: D102, ANN201
        pass

    @abc.abstractmethod
    def _write_slide(self) -> np.ndarray:
        pass

    def write_slide(self) -> None:
        """Write a pyramid slide."""
        with ProcessManager.process(f"{self.base_path} - {self.output_depth}"):
            ProcessManager.submit_thread(self._write_slide)

            ProcessManager.join_threads()

    def scale_info(self, scale: int) -> dict:
        """Return scale information for a given scale."""
        if scale == -1:
            return self.info["scales"][0]

        scale_info = None

        for res in self.info["scales"]:
            if int(res["key"]) == scale:
                scale_info = res
                break

        if scale_info is None:
            ValueError(f"No scale information for resolution {scale}.")

        return scale_info  # type: ignore

    def store_chunk(
        self,
        image: np.ndarray,
        key: str,
        chunk_coords: tuple[int, ...],
    ) -> None:
        """Store a pyramid chunk.

        Inputs:
            image: byte stream to save to disk
            key: pyramid scale, folder to save chunk to
            chunk_coords: X,Y,Z coordinates of data in buf
        """
        buf = self.encoder.encode(image)

        self._write_chunk(key, chunk_coords, buf)

    def _chunk_path(
        self,
        key: str,
        chunk_coords: tuple[int, ...],
        pattern: typing.Optional[str] = None,
    ) -> pathlib.Path:
        if pattern is None:
            pattern = self.chunk_pattern
        chunk_coords = self._chunk_coords(chunk_coords)
        chunk_filename = pattern.format(*chunk_coords, key=key)
        return self.base_path.joinpath(chunk_filename)

    def _chunk_coords(
        self,
        chunk_coords: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(chunk_coords) == 4:  # noqa: PLR2004
            chunk_coords = (*chunk_coords, self.output_depth, self.output_depth + 1)
        elif len(chunk_coords) != 6:  # noqa: PLR2004
            msg = "chunk_coords must be a 4-tuple or a 6-tuple."
            raise ValueError(msg)
        return chunk_coords


def _get_higher_res(
    scale: int,
    slide_writer: PyramidWriter,
    x_: typing.Optional[tuple[int, int]] = None,
    y_: typing.Optional[tuple[int, int]] = None,
    z_range: tuple[int, int] = (0, 1),
) -> np.ndarray:
    r"""Recursive function for pyramid building.

    This is a recursive function that builds an image pyramid by indicating
    an original region of an image at a given scale. This function then
    builds a pyramid up from the highest resolution components of the pyramid
    (the original images) to the given position resolution.

    As an example, imagine the following possible pyramid:

    Scale S=0                     1234
                                 /    \
    Scale S=1                  12      34
                              /  \    /  \
    Scale S=2                1    2  3    4

    At scale 2 (the highest resolution) there are 4 original images. At scale 1,
    images are averaged and concatenated into one image (i.e. image 12). Calling
    this function using S=0 will attempt to generate 1234 by calling this
    function again to get image 12, which will then call this function again to
    get image 1 and then image 2. Note that this function actually builds images
    in quadrants (top left and right, bottom left and right) rather than two
    sections as displayed above.

    Due to the nature of how this function works, it is possible to build a
    pyramid in parallel, since building the subpyramid under image 12 can be run
    independently of the building of subpyramid under 34.

    Args:
        scale: Top level scale from which the pyramid will be built
        slide_writer: object used to encode and write pyramid tiles
        x_: Range of X values [min,max] to get at the indicated scale
        y_: Range of Y values [min,max] to get at the indicated scale
        z_range: Range of Z values [min,max]

    Returns:
        image: The image corresponding to the X,Y values at scale S
    """
    # Get the scale info
    scale_info = slide_writer.scale_info(scale)

    x_range = x_ if x_ is not None else [0, scale_info["size"][0]]
    y_range = y_ if y_ is not None else [0, scale_info["size"][1]]

    # Modify upper bound to stay within resolution dimensions
    if x_range[1] > scale_info["size"][0]:
        x_range[1] = scale_info["size"][0]  # type: ignore
    if y_range[1] > scale_info["size"][1]:
        y_range[1] = scale_info["size"][1]  # type: ignore

    if str(scale) == slide_writer.scale_info(-1)["key"]:
        with (
            ProcessManager.thread(),
            bfio.BioReader(slide_writer.image_path, max_workers=1) as br,
        ):
            image = br[
                y_range[0] : y_range[1],
                x_range[0] : x_range[1],
                z_range[0] : z_range[1],
                ...,
            ].squeeze()

        # Write the chunk
        slide_writer.store_chunk(
            image,
            str(scale),
            (x_range[0], x_range[1], y_range[0], y_range[1]),
        )

        return image

    # Initialize the output
    image = np.zeros(
        (y_range[1] - y_range[0], x_range[1] - x_range[0]),
        dtype=slide_writer.dtype,
    )

    # Set the subgrid dimensions
    subgrid_dims = [[2 * x_range[0], 2 * x_range[1]], [2 * y_range[0], 2 * y_range[1]]]
    for dim in subgrid_dims:
        while dim[1] - dim[0] > CHUNK_SIZE:
            dim.insert(
                1,
                dim[0] + ((dim[1] - dim[0] - 1) // CHUNK_SIZE) * CHUNK_SIZE,
            )

    def load_and_scale(*args, **kwargs) -> None:  # noqa ANN002, ANN003
        sub_image = _get_higher_res(**kwargs)

        with ProcessManager.thread():
            image, x_ind, y_ind = args[:3]
            image[y_ind[0] : y_ind[1], x_ind[0] : x_ind[1]] = kwargs[
                "slide_writer"
            ].scale(sub_image)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for y in range(0, len(subgrid_dims[1]) - 1):
            y_ind = [
                subgrid_dims[1][y] - subgrid_dims[1][0],
                subgrid_dims[1][y + 1] - subgrid_dims[1][0],
            ]
            y_ind = [np.ceil(yi / 2).astype("int") for yi in y_ind]
            for x in range(0, len(subgrid_dims[0]) - 1):
                x_ind = [
                    subgrid_dims[0][x] - subgrid_dims[0][0],
                    subgrid_dims[0][x + 1] - subgrid_dims[0][0],
                ]
                x_ind = [np.ceil(xi / 2).astype("int") for xi in x_ind]
                futures.append(
                    executor.submit(
                        load_and_scale,
                        image,
                        x_ind,
                        y_ind,  # args
                        scale=scale + 1,
                        slide_writer=slide_writer,
                        x_=subgrid_dims[0][x : x + 2],  # kwargs
                        y_=subgrid_dims[1][y : y + 2],
                        z_range=z_range,
                    ),
                )
            for f in concurrent.futures.as_completed(futures):
                f.result()

    # Write the chunk
    slide_writer.store_chunk(
        image,
        str(scale),
        (x_range[0], x_range[1], y_range[0], y_range[1]),
    )
    return image


class NeuroglancerWriter(PyramidWriter):
    """Method to write a Neuroglancer pre-computed pyramid.

    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.chunk_pattern = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"

        min_level = min([int(self.scale_info(-1)["key"]), 10])
        self.info = bfio_metadata_to_slide_info(
            self.image_path,
            self.max_output_depth,
            self.image_type,
            min_level,
        )

        if self.image_type == "segmentation":
            self.labels = set()  # type: ignore

    def store_chunk(
        self,
        image: np.ndarray,
        key: str,
        chunk_coords: tuple[int, ...],
    ) -> None:
        """Store a chunk of data in the pyramid."""
        # Add in a label aggregator to the store_chunk operation
        # Only aggregate labels at the highest resolution
        if self.image_type == "segmentation":
            if key == self.scale_info(-1)["key"]:
                self.labels = self.labels.union(set(np.unique(image)))
            elif key == self.info["scales"][-1]["key"]:
                root = zarr.open(str(self.base_path.joinpath("labels.zarr")))
                if str(self.output_depth) not in root.array_keys():
                    labels = root.empty(
                        str(self.output_depth),
                        shape=(len(self.labels),),
                        dtype=np.uint64,
                    )
                else:
                    labels = root[str(self.output_depth)]
                labels[:] = np.asarray(list(self.labels), np.uint64).squeeze()

        super().store_chunk(image, key, chunk_coords)

    def _write_chunk(
        self,
        key: str,
        chunk_coords: tuple[int, ...],
        buf: np.array,
    ) -> None:
        chunk_path = self._chunk_path(key, chunk_coords)
        chunk_path.parent.mkdir(exist_ok=True, parents=True)
        with chunk_path.with_name(chunk_path.name).open("wb") as f:
            f.write(buf)

    def _encoder(self) -> "ChunkEncoder":
        return NeuroglancerChunkEncoder(self.info)

    def _write_slide(self) -> np.ndarray:
        pathlib.Path(self.base_path).mkdir(exist_ok=True)

        # Don't create a full pyramid to help reduce bounding box size
        start_level = int(self.info["scales"][-1]["key"])
        _get_higher_res(
            start_level,
            self,
            z_range=(self.image_depth, self.image_depth + 1),
        )

    def write_info(self) -> None:
        """This creates the info file specifying the metadata for the format."""
        # Create an output path object for the info file
        op = pathlib.Path(self.base_path)
        op.mkdir(exist_ok=True, parents=True)
        op = op.joinpath("info")

        # Write the neuroglancer info file
        with op.open("w") as writer:
            json.dump(self.info, writer, indent=2)

        if self.image_type == "segmentation":
            self._write_segment_info()

    def _write_segment_info(self) -> None:
        """This function creates the info file needed to segment the image."""
        if self.image_type != "segmentation":
            msg = (
                "The NeuroglancerWriter object must have "
                'image_type = "segmentation" to use write_segment_info.'
            )
            raise TypeError(
                msg,
            )

        op = pathlib.Path(self.base_path).joinpath("infodir")
        op.mkdir(exist_ok=True)
        op = op.joinpath("info")

        # Get the labels
        root = zarr.open(str(self.base_path.joinpath("labels.zarr")))
        labels = set()  # type: ignore
        for d in root.array_keys():
            labels = labels.union(set(root[d][:].squeeze().tolist()))

        inlineinfo = {
            "ids": [str(item) for item in labels],
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": [str(item) for item in labels],
                },
                {
                    "id": "description",
                    "type": "label",
                    "values": [str(item) for item in labels],
                },
            ],
        }

        info = {"@type": "neuroglancer_segment_properties", "inline": inlineinfo}

        # writing all the information into the file
        with op.open("w") as writer:
            json.dump(info, writer, indent=2)


class ZarrWriter(PyramidWriter):
    """Method to write a Zarr pyramid.

    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the ZarrWriter object."""
        super().__init__(*args, **kwargs)

        out_name = self.base_path.name.replace("".join(self.base_path.suffixes), "")
        self.base_path = self.base_path.with_name(out_name)
        self.base_path.mkdir(exist_ok=True)
        self.root = zarr.open(
            str(self.base_path.joinpath("data.zarr").resolve()),
            mode="a",
        )
        if "0" in self.root.group_keys():
            self.root = self.root["0"]
        else:
            self.root = self.root.create_group("0")

        self.writers = {}
        max_scale = int(self.scale_info(-1)["key"])
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        for scale in range(len(self.info["scales"])):
            scale_info = self.scale_info(scale)
            key = str(max_scale - int(scale_info["key"]))
            if key not in self.root.array_keys():
                self.writers[key] = self.root.zeros(
                    key,
                    shape=(
                        1,
                        self.max_output_depth,
                        1,
                        scale_info["size"][1],
                        scale_info["size"][0],
                    ),
                    chunks=(1, 1, 1, CHUNK_SIZE, CHUNK_SIZE),
                    dtype=self.dtype,
                    compressor=compressor,
                )
            else:
                self.root[key].resize(
                    (
                        1,
                        self.max_output_depth,
                        1,
                        scale_info["size"][1],
                        scale_info["size"][0],
                    ),
                )
                self.writers[key] = self.root[key]

    def _write_chunk(
        self,
        key: str,
        chunk_coords: tuple[int, ...],
        buf: np.array,
    ) -> None:
        key = str(int(self.scale_info(-1)["key"]) - int(key))  # type: ignore
        chunk_coords = self._chunk_coords(chunk_coords)

        self.writers[key][  # type: ignore
            0:1,
            chunk_coords[4] : chunk_coords[5],
            0:1,
            chunk_coords[2] : chunk_coords[3],
            chunk_coords[0] : chunk_coords[1],
        ] = buf

    def _encoder(self) -> "ChunkEncoder":
        return ZarrChunkEncoder(self.info)

    def _write_slide(self) -> np.ndarray:
        _get_higher_res(0, self, z_range=(self.image_depth, self.image_depth + 1))

    def write_info(self) -> None:
        """This creates the multiscales metadata for zarr pyramids."""
        # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
        multi_scales = [
            {
                "version": "0.1",
                "name": self.base_path.name,
                "datasets": [],
                "metadata": {"method": "mean"},
            },
        ]

        len(self.scale_info(-1)["key"])
        max_scale = int(self.scale_info(-1)["key"])
        for scale in reversed(range(len(self.info["scales"]))):
            scale_info = self.scale_info(scale)
            key = str(max_scale - int(scale_info["key"]))
            multi_scales[0]["datasets"].append({"path": key})  # type: ignore
        self.root.attrs["multiscales"] = multi_scales

        with bfio.BioReader(self.image_path, max_workers=1) as bfio_reader:
            metadata = OMEXML(str(bfio_reader.metadata))
            metadata.image(0).Pixels.SizeC = self.max_output_depth
            metadata.image(0).Pixels.channel_count = self.max_output_depth

            for c in range(self.max_output_depth):  # type: ignore
                metadata.image().Pixels.Channel(c).Name = f"Channel {c}"

            with self.base_path.joinpath("METADATA.ome.xml").open("x") as fw:
                fw.write(str(metadata).replace("<ome:", "<").replace("</ome:", "</"))


class DeepZoomWriter(PyramidWriter):
    """Method to write a DeepZoom pyramid.

    Inputs:
        base_dir: Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the DeepZoomWriter."""
        super().__init__(*args, **kwargs)
        self.chunk_pattern = "{key}/{0}_{1}.png"
        self.base_path = self.base_path.joinpath(str(self.output_depth) + "_files")

    def _chunk_coords(self, chunk_coords: tuple[int, ...]) -> tuple[int, int]:
        return (chunk_coords[0] // CHUNK_SIZE, chunk_coords[2] // CHUNK_SIZE)

    def _write_chunk(
        self,
        key: str,
        chunk_coords: tuple[int, ...],
        buf: np.array,
    ) -> None:
        chunk_path = self._chunk_path(key, chunk_coords)
        chunk_path.parent.mkdir(exist_ok=True, parents=True)
        imageio.imwrite(
            str(chunk_path.with_name(chunk_path.name)),
            buf,
            format="PNG-FI",
            compression=1,
        )

    def write_info(self) -> None:
        """This creates the info file for a DeepZoom pyramid."""
        # Create an output path object for the info file
        op = pathlib.Path(self.base_path).parent.joinpath(
            f"{self.output_depth}.dzi",
        )

        # DZI file template
        dzi = (
            '<?xml version="1.0" encoding="utf-8"?><Image TileSize="{}" '
            'Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/'
            'deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'
        )

        # write the dzi file
        with op.open("w") as writer:
            writer.write(
                dzi.format(
                    CHUNK_SIZE,
                    self.info["scales"][0]["size"][0],
                    self.info["scales"][0]["size"][1],
                ),
            )

    def _write_slide(self) -> np.ndarray:
        pathlib.Path(self.base_path).mkdir(exist_ok=False)

        _get_higher_res(0, self, z_range=(self.image_depth, self.image_depth + 1))

    def _encoder(self) -> "ChunkEncoder":
        return DeepZoomChunkEncoder(self.info)

    def write_segment_info(self) -> None:
        """DeepZoom does not have a segmentation format."""
        msg = "DeepZoom does not have a segmentation format."
        raise NotImplementedError(msg)


class ChunkEncoder:
    """Base class for chunk encoders.

    Modified and condensed from multiple functions and classes
    https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/chunk_encoding.py
    """

    # Data types used by Neuroglancer
    DATA_TYPES = (
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
    )

    def __init__(self, info: dict) -> None:
        """Initialize the encoder.

        Inputs:
            info: dict containing the following keys:
                data_type: string, one of ChunkEncoder.DATA_TYPES
                num_channels: int, number of channels in the data
        """
        try:
            data_type = info["data_type"]
            num_channels = info["num_channels"]
        except KeyError as exc:
            msg = f"The info dict is missing an essential key {exc}"
            raise KeyError(msg) from exc

        if not isinstance(num_channels, int) or not num_channels > 0:
            msg = (
                f"Invalid value {num_channels} for num_channels "
                f"(must be a positive integer)"
            )
            raise KeyError(msg)

        if data_type not in ChunkEncoder.DATA_TYPES:
            msg = (
                f"Invalid data_type {data_type} "
                f"(should be one of {ChunkEncoder.DATA_TYPES})"
            )
            raise KeyError(msg)

        self.info = info
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")

    @abc.abstractmethod
    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode a chunk from a Numpy array into bytes."""
        pass


class NeuroglancerChunkEncoder(ChunkEncoder):
    """Encoder for Neuroglancer chunks."""

    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode a chunk from a Numpy array into bytes.

        Args:
            chunk: array with 2 dimensions

        Returns:
            encoded chunk (byte stream).
        """
        # Rearrange the image for Neuroglancer
        chunk = np.moveaxis(
            chunk.reshape(chunk.shape[0], chunk.shape[1], 1, 1),
            (0, 1, 2, 3),
            (2, 3, 1, 0),
        )
        chunk = np.asarray(chunk).astype(self.dtype)
        assert chunk.ndim == 4  # noqa S101
        assert chunk.shape[0] == self.num_channels  # noqa S101
        return chunk.tobytes()


class ZarrChunkEncoder(ChunkEncoder):
    """Encoder for Zarr chunks."""

    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode a chunk from a Numpy array into bytes.

        Args:
            chunk: array with 2 dimensions

        Returns:
            encoded chunk (byte stream).
        """
        # Rearrange the image for Neuroglancer
        chunk = chunk.reshape(chunk.shape[0], chunk.shape[1], 1, 1, 1).transpose(
            4,
            2,
            3,
            0,
            1,
        )
        return np.asarray(chunk).astype(self.dtype)


class DeepZoomChunkEncoder(ChunkEncoder):
    """Encoder for DeepZoom chunks."""

    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode a chunk for DeepZoom.

        Nothing special to do for encoding except checking the number of
        dimensions.

        Args:
            chunk: array with 2 dimensions

        Returns:
            encoded chunk (byte stream)
        """
        # Check to make sure the data is formatted properly
        assert chunk.ndim == 2  # noqa: PLR2004, S101
        return chunk


def bfio_metadata_to_slide_info(
    image_path: pathlib.Path,
    stack_height: int,
    image_type: str,
    min_scale: int = 0,
) -> dict[str, typing.Any]:
    """Generate a Neuroglancer info file from Bioformats metadata.

    Neuroglancer requires an info file in the root of the pyramid directory.
    All information necessary for this info file is contained in Bioformats
    metadata, so this function takes the metadata and generates the info file.

    Args:
        image_path: Path to the image file
        out_path: Path to directory where pyramid will be generated
        stack_height: Number of images to stack together
        image_type: Type of image (e.g. "image", "segmentation")
        min_scale: Minimum scale to generate (default 0)

    Returns:
        info: A dictionary containing the information in the info file
    """
    with bfio.BioReader(image_path, max_workers=1) as bfio_reader:
        # Get metadata info from the bfio reader
        sizes = [bfio_reader.X, bfio_reader.Y, stack_height]

        phys_x = bfio_reader.ps_x
        if None in phys_x:
            phys_x = (1000, "nm")

        phys_y = bfio_reader.ps_y
        if None in phys_y:
            phys_y = (1000, "nm")

        phys_z = bfio_reader.ps_z
        if None in phys_z:
            phys_z = ((phys_x[0] + phys_y[0]) / 2, phys_x[1])

        resolution = [phys_x[0] * UNITS[phys_x[1]]]
        resolution.append(phys_y[0] * UNITS[phys_y[1]])
        resolution.append(phys_z[0] * UNITS[phys_z[1]])  # Just used as a placeholder
        dtype = str(np.dtype(bfio_reader.dtype))

    num_scales = int(np.ceil(np.log2(max(sizes))))

    # create a scales template, use the full resolution8
    scales = {
        "chunk_sizes": [[CHUNK_SIZE, CHUNK_SIZE, 1]],
        "encoding": "raw",
        "key": str(num_scales),
        "resolution": resolution,
        "size": sizes,
        "voxel_offset": [0, 0, 0],
    }

    # initialize the json dictionary
    info = {
        "data_type": dtype,
        "num_channels": 1,
        "scales": [scales],
        "type": image_type,
    }

    if image_type == "segmentation":
        info["segment_properties"] = "infodir"

    for i in reversed(range(min_scale, num_scales)):
        previous_scale = info["scales"][-1]  # type: ignore
        current_scale = copy.deepcopy(previous_scale)
        current_scale["key"] = str(i)
        current_scale["size"] = [
            int(np.ceil(previous_scale["size"][0] / 2)),
            int(np.ceil(previous_scale["size"][1] / 2)),
            stack_height,
        ]
        current_scale["resolution"] = [
            2 * previous_scale["resolution"][0],
            2 * previous_scale["resolution"][1],
            previous_scale["resolution"][2],
        ]
        info["scales"].append(current_scale)  # type: ignore

    return info
