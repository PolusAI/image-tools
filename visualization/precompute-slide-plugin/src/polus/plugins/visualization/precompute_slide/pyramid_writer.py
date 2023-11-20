"""Provides the PyramidWriter classes."""

import abc
import json
import logging
import pathlib
import typing

import bfio
import imageio
import numcodecs
import numpy
import preadator
import zarr

from . import chunk_encoder
from . import utils

logger = logging.getLogger(__file__)
logger.setLevel(utils.POLUS_LOG)


class PyramidWriter(abc.ABC):
    """Pyramid file writing base class.

    Modified and condensed from FileAccessor class in neuroglancer-scripts
    https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/file_accessor.py

    This class should not be called directly. It should be inherited by a pyramid
    writing class type.
    """

    chunk_pattern: str

    def __init__(  # noqa: PLR0913
        self,
        base_dir: typing.Union[pathlib.Path, str],
        image_path: typing.Union[pathlib.Path, str],
        image_depth: int = 0,
        output_depth: int = 0,
        max_output_depth: typing.Optional[int] = None,
        image_type: utils.ImageType = utils.ImageType.Intensity,
    ) -> None:
        """Initialize the pyramid writer.

        Inputs:
            base_dir - Where pyramid folders and info file will be stored
            image_path - Path to image
            image_depth - Z index of image to use
            output_depth - Z index of image to write
            max_output_depth - Maximum Z index of image to write
            image_type - Type of image to write
        """
        if isinstance(image_path, str):
            image_path = pathlib.Path(image_path)
        self.image_path = image_path
        if isinstance(base_dir, str):
            base_dir = pathlib.Path(base_dir)
        self.base_path = base_dir
        self.image_depth = image_depth
        self.output_depth = output_depth
        self.max_output_depth = max_output_depth or 128
        self.image_type = image_type

        if image_type == utils.ImageType.Intensity:
            self.scale = utils._avg2
        elif image_type == utils.ImageType.Segmentation:
            self.scale = utils._mode2
        else:
            msg = 'image_type must be one of ["image","segmentation"]'
            raise ValueError(msg)

        self.info = utils.bfio_metadata_to_slide_info(
            self.image_path,
            self.base_path,
            self.max_output_depth,
            self.image_type,
        )

        self.dtype = self.info["data_type"]

        self.encoder = self._encoder()

    @abc.abstractmethod
    def _encoder(self) -> chunk_encoder.ChunkEncoder:
        """Return the associated encoder."""
        pass

    @abc.abstractmethod
    def _write_chunk(
        self,
        key: str,
        chunk_path: pathlib.Path,
        buf: numpy.ndarray,
    ) -> None:
        """Write a chunk to disk."""
        pass

    @abc.abstractmethod
    def write_info(self) -> None:
        """Write pyramid information."""
        pass

    # @abc.abstractmethod
    # def write_segment_info(self) -> None:
    #     """Write segmentation information."""
    #     pass

    def write_slide(self) -> None:
        """Write the slide."""
        with preadator.ProcessManager(name="write_slide") as pm:
            logger.debug(f"submitting process for writing slide {self.image_path}")
            pm.submit_process(self._write_slide)

    @abc.abstractmethod
    def _write_slide(self) -> None:
        """Write the slide."""
        pass

    def scale_info(self, scale: int) -> dict:
        """Return the scale information for a given scale."""
        if scale == -1:
            return self.info["scales"][0]

        scale_info: dict

        for res in self.info["scales"]:
            if int(res["key"]) == scale:
                scale_info = res
                break
        else:
            ValueError(f"No scale information for resolution {scale}.")

        return scale_info

    def store_chunk(
        self,
        image: bytes,
        key: str,
        chunk_coords: tuple[int, ...],
    ) -> None:
        """Store a pyramid chunk.

        Inputs:
            image: byte stream to save to disk
            key: pyramid scale, folder to save chunk to
            chunk_coords: X,Y,Z coordinates of data in buf
        """
        logger.debug(f"storing chunk {key = } {chunk_coords = }")
        buf = self.encoder.encode(image)

        # TODO: type of chunk_coords may be broken here
        self._write_chunk(key, chunk_coords, buf)  # type: ignore[arg-type]

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
        return self.base_path / chunk_filename

    def _chunk_coords(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        if len(chunk_coords) == 4:  # noqa: PLR2004
            chunk_coords = (*chunk_coords, self.output_depth, self.output_depth + 1)
        elif len(chunk_coords) != 6:  # noqa: PLR2004
            msg = "chunk_coords must be a 4-tuple or a 6-tuple."
            raise ValueError(msg)
        return chunk_coords


def _get_higher_res(
    scale: int,
    slide_writer: PyramidWriter,
    x: typing.Optional[tuple[int, int]] = None,
    y: typing.Optional[tuple[int, int]] = None,
    z: tuple[int, int] = (0, 1),
) -> numpy.ndarray:
    r"""Recursive function for pyramid building.

    This is a recursive function that builds an image pyramid by indicating
    an original region of an image at a given scale. This function then
    builds a pyramid up from the highest resolution components of the pyramid
    (the original images) to the given position resolution.

    As an example, imagine the following possible pyramid:

    Scale S=0                     1234
                                /      \
    Scale S=1                 12        34
                             /  \      /  \
    Scale S=2               1    2    3    4

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
        file_path: Path to image
        slide_writer: object used to encode and write pyramid tiles
        x: Range of X values [min,max] to get at the indicated scale
        y: Range of Y values [min,max] to get at the indicated scale
        z: Range of Z values [min,max] to get at the indicated scale
    Returns:
        image: The image corresponding to the X,Y values at scale S
    """
    logger.debug(
        f"getting higher res for scale {scale} in {slide_writer.__class__.__name__} for {slide_writer.image_path} with coordinates {x = }, {y = }, {z = }",  # noqa: E501
    )

    # Get the scale info
    scale_info = slide_writer.scale_info(scale)

    if x is None:
        x = (0, scale_info["size"][0])
    if y is None:
        y = (0, scale_info["size"][1])

    x_min, x_max = x
    y_min, y_max = y
    z_min, z_max = z

    # Modify upper bound to stay within resolution dimensions
    if x_max > scale_info["size"][0]:
        x_max = scale_info["size"][0]
    if y_max > scale_info["size"][1]:
        y_max = scale_info["size"][1]

    if str(scale) == slide_writer.scale_info(-1)["key"]:
        logger.info("nested process manager...")

        with bfio.BioReader(slide_writer.image_path, max_workers=1) as br:
            image = br[y_min:y_max, x_min:x_max, z_min:z_max, ...].squeeze()

        # Write the chunk
        slide_writer.store_chunk(image, str(scale), (x_min, x_max, y_min, y_max))

        return image

    # Initialize the output
    image = numpy.zeros((y_max - y_min, x_max - x_min), dtype=slide_writer.dtype)

    # Set the subgrid dimensions
    subgrid_dims = [[2 * x_min, 2 * x_max], [2 * y_min, 2 * y_max]]
    for dim in subgrid_dims:
        while dim[1] - dim[0] > utils.CHUNK_SIZE:
            dim.insert(
                1,
                dim[0] + ((dim[1] - dim[0] - 1) // utils.CHUNK_SIZE) * utils.CHUNK_SIZE,
            )

    def load_and_scale(*args, **kwargs) -> None:  # noqa: ANN002 ANN003
        logger.debug(f"loading and scaling {args = }, {kwargs = }")
        sub_image = _get_higher_res(**kwargs)

        image = args[0]
        x_ind = args[1]
        y_ind = args[2]
        image[y_ind[0] : y_ind[1], x_ind[0] : x_ind[1]] = kwargs["slide_writer"].scale(
            sub_image,
        )

    # with preadator.ProcessManager(name="get_higher_res") as pm:
    for y_ in range(0, len(subgrid_dims[1]) - 1):
        y_ind = [
            subgrid_dims[1][y_] - subgrid_dims[1][0],
            subgrid_dims[1][y_ + 1] - subgrid_dims[1][0],
        ]
        y_ind = [numpy.ceil(yi / 2).astype("int") for yi in y_ind]
        for x_ in range(0, len(subgrid_dims[0]) - 1):
            x_ind = [
                subgrid_dims[0][x_] - subgrid_dims[0][0],
                subgrid_dims[0][x_ + 1] - subgrid_dims[0][0],
            ]
            x_ind = [numpy.ceil(xi / 2).astype("int") for xi in x_ind]
            # pm.submit_thread(
            load_and_scale(
                image,
                x_ind,
                y_ind,
                x=subgrid_dims[0][x_ : x_ + 2],
                y=subgrid_dims[1][y_ : y_ + 2],
                z=z,
                scale=scale + 1,
                slide_writer=slide_writer,
            )

    # Write the chunk
    slide_writer.store_chunk(image, str(scale), (x_min, x_max, y_min, y_max))
    return image


class NeuroglancerWriter(PyramidWriter):
    """Method to write a Neuroglancer pre-computed pyramid.

    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002 ANN003
        """Initialize the pyramid writer."""
        super().__init__(*args, **kwargs)
        self.chunk_pattern = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"

        min_level = min([int(self.scale_info(-1)["key"]), 10])
        self.info = utils.bfio_metadata_to_slide_info(
            self.image_path,
            self.base_path,
            self.max_output_depth,
            self.image_type,
            min_level,
        )

        if self.image_type == utils.ImageType.Segmentation:
            self.labels: set[int] = set()

    def store_chunk(
        self,
        image: numpy.ndarray,
        key: str,
        chunk_coords: tuple[int, ...],
    ) -> None:
        """Store a pyramid chunk."""
        # Add in a label aggregator to the store_chunk operation
        # Only aggregate labels at the highest resolution
        if self.image_type == utils.ImageType.Segmentation:
            if key == self.scale_info(-1)["key"]:
                self.labels = self.labels.union(set(numpy.unique(image)))
            elif key == self.info["scales"][-1]["key"]:
                root = zarr.open(str(self.base_path.joinpath("labels.zarr")))
                if str(self.output_depth) not in root.array_keys():
                    labels = root.empty(
                        str(self.output_depth),
                        shape=(len(self.labels),),
                        dtype=numpy.uint64,
                    )
                else:
                    labels = root[str(self.output_depth)]
                labels[:] = numpy.asarray(list(self.labels), numpy.uint64).squeeze()

        super().store_chunk(image, key, chunk_coords)

    def _write_chunk(
        self,
        key: str,
        chunk_coords: tuple[int, ...],  # type: ignore[override]
        buf: bytes,
    ) -> None:
        chunk_path = self._chunk_path(key, chunk_coords)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        with chunk_path.with_name(chunk_path.name).open("wb") as f:
            f.write(buf)

    def _encoder(self) -> chunk_encoder.NeuroglancerChunkEncoder:
        return chunk_encoder.NeuroglancerChunkEncoder(self.info)

    def _write_slide(self) -> None:
        pathlib.Path(self.base_path).mkdir(exist_ok=True)

        # Don't create a full pyramid to help reduce bounding box size
        start_level = int(self.info["scales"][-1]["key"])
        _get_higher_res(
            start_level,
            self,
            z=(self.image_depth, self.image_depth + 1),
        )

    def write_info(self) -> None:
        """Creates the info file for the metadata for the precomputed format."""
        # Create an output path object for the info file
        op = pathlib.Path(self.base_path)
        op.mkdir(exist_ok=True, parents=True)
        op = op.joinpath("info")

        # Write the neuroglancer info file
        with op.open("w") as writer:
            json.dump(self.info, writer, indent=2)

        if self.image_type == utils.ImageType.Segmentation:
            self._write_segment_info()

    def _write_segment_info(self) -> None:
        """This function creates the info file needed to segment the image."""
        if self.image_type != utils.ImageType.Segmentation:
            msg = 'The NeuroglancerWriter object must have image_type = "segmentation" to use write_segment_info.'  # noqa: E501
            raise TypeError(msg)

        op = pathlib.Path(self.base_path).joinpath("infodir")
        op.mkdir(exist_ok=True)
        op = op.joinpath("info")

        # Get the labels
        root = zarr.open(str(self.base_path.joinpath("labels.zarr")))
        labels: set[str] = set()
        for d in root.array_keys():
            labels = labels.union(map(str, root[d][:].squeeze().tolist()))

        ids = list(map(str, labels))

        inlineinfo = {
            "ids": ids,
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": ids,
                },
                {
                    "id": "description",
                    "type": "label",
                    "values": ids,
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

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002 ANN003
        """Initialize the pyramid writer."""
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

        self.writers: dict[str, typing.Union[zarr.Group, zarr.Array]] = {}
        max_scale = int(self.scale_info(-1)["key"])
        compressor = numcodecs.Blosc(
            cname="zstd",
            clevel=3,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
        )
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
                    chunks=(1, 1, 1, utils.CHUNK_SIZE, utils.CHUNK_SIZE),
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
        chunk_coords: tuple[int, ...],  # type: ignore[override]
        buf: numpy.ndarray,
    ) -> None:
        """Write a chunk to disk."""
        key = str(int(self.scale_info(-1)["key"]) - int(key))
        chunk_coords = self._chunk_coords(chunk_coords)

        self.writers[key][
            0:1,
            chunk_coords[4] : chunk_coords[5],
            0:1,
            chunk_coords[2] : chunk_coords[3],
            chunk_coords[0] : chunk_coords[1],
        ] = buf

    def _encoder(self) -> chunk_encoder.ZarrChunkEncoder:
        """Return the associated chunk encoder."""
        return chunk_encoder.ZarrChunkEncoder(self.info)

    def _write_slide(self) -> None:
        """Write the slide."""
        _get_higher_res(0, self, z=(self.image_depth, self.image_depth + 1))

    def write_info(self) -> None:
        """This creates the multiscales metadata for zarr pyramids."""
        # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
        multiscales = [
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
            multiscales[0]["datasets"].append({"path": key})  # type: ignore[attr-defined]  # noqa: E501
        self.root.attrs["multiscales"] = multiscales

        with bfio.BioReader(self.image_path, max_workers=1) as bfio_reader:
            metadata = bfio.OmeXml.OMEXML(str(bfio_reader.metadata))
            metadata.image(0).Pixels.SizeC = self.max_output_depth
            metadata.image(0).Pixels.channel_count = self.max_output_depth

            for c in range(self.max_output_depth):
                metadata.image().Pixels.Channel(c).Name = f"Channel {c}"

            try:
                file = self.base_path.joinpath("METADATA.ome.xml").open("x")
            except FileExistsError:
                file = self.base_path.joinpath("METADATA.ome.xml").open("w")
                logger.warning(f"file already exists : {file.name}. Overwriting.")
            finally:
                file.write(str(metadata).replace("<ome:", "<").replace("</ome:", "</"))
                file.close()


class DeepZoomWriter(PyramidWriter):
    """Method to write a DeepZoom pyramid.

    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002 ANN003
        """Initialize the pyramid writer."""
        super().__init__(*args, **kwargs)
        self.chunk_pattern = "{key}/{0}_{1}.png"
        self.base_path = self.base_path.joinpath(str(self.output_depth) + "_files")

    def _chunk_coords(self, chunk_coords: tuple[int, ...]) -> tuple[int, int]:
        """Convert chunk coordinates to DeepZoom coordinates."""
        return (
            chunk_coords[0] // utils.CHUNK_SIZE,
            chunk_coords[2] // utils.CHUNK_SIZE,
        )

    def _write_chunk(
        self,
        key: str,
        chunk_coords: tuple[int, ...],  # type: ignore[override]
        buf: numpy.ndarray,
    ) -> None:
        """Write a chunk to disk."""
        chunk_path = self._chunk_path(key, chunk_coords)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(
            str(chunk_path.with_name(chunk_path.name)),
            buf,
            format="PNG-FI",
            compression=1,
        )

    def write_info(self) -> None:
        """Write the info file."""
        # Create an output path object for the info file
        op = pathlib.Path(self.base_path).parent.joinpath(
            f"{self.output_depth}.dzi",
        )

        # DZI file template
        dzi_template = '<?xml version="1.0" encoding="utf-8"?><Image TileSize="{}" Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'  # noqa: E501

        # write the dzi file
        with op.open("w") as writer:
            writer.write(
                dzi_template.format(
                    utils.CHUNK_SIZE,
                    self.info["scales"][0]["size"][0],
                    self.info["scales"][0]["size"][1],
                ),
            )

    def _write_slide(self) -> None:
        """Write the slide."""
        pathlib.Path(self.base_path).mkdir(exist_ok=False)

        _get_higher_res(0, self, z=(self.image_depth, self.image_depth + 1))

    def _encoder(self) -> chunk_encoder.DeepZoomChunkEncoder:
        """Return the associated chunk encoder."""
        return chunk_encoder.DeepZoomChunkEncoder(self.info)

    def write_segment_info(self) -> None:
        """Write the segmentation info."""
        msg = "DeepZoom does not have a segmentation format."
        raise NotImplementedError(msg)
