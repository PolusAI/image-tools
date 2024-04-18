"""Omero Download Tool."""

import logging
import os
import shutil
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from bfio import BioWriter
from omero.gateway import BlitzGateway
from omero.plugins.download import DownloadControl
from pydantic import BaseModel as V2BaseModel
from pydantic import model_validator

OMERO_USERNAME = os.environ.get("OMERO_USERNAME")
OMERO_PASSWORD = os.environ.get("OMERO_PASSWORD")
HOST = "165.112.226.159"
PORT = 4064


logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def generate_preview(
    path: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    source_path = Path().cwd().parents[4].joinpath("examples")
    shutil.copytree(source_path, path, dirs_exist_ok=True)


class DATATYPE(str, Enum):
    """Objects types."""

    PROJECT = "project"
    DATASET = "dataset"
    SCREEN = "screen"
    PLATE = "plate"
    WELL = "well"
    Default = "dataset"


class ServerConnection(V2BaseModel):
    """Establishes a connection to an OMERO server using BlitzGateway.

    Args:
        username:  The username for authentication.
        password:  The password for authentication
        host: The IP address of the OMERO server.
        port: Port used to establish a connection between client and server.

    Returns:
         BlitzGateway: A connection object to the OMERO server.
    """

    username: str
    password: str
    host: str
    port: int

    def _authentication(self) -> BlitzGateway:
        """Connection to an OMERO server using BlitzGateway."""
        return BlitzGateway(
            self.username,
            self.password,
            host=self.host,
            port=self.port,
        )


class CustomValidation(V2BaseModel):
    """Properties with validation."""

    data_type: str
    out_dir: Path
    name: Optional[str] = None
    object_id: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, values: Any) -> Any:  # noqa: ANN401
        """Validation of Paths."""
        out_dir = values.get("out_dir")
        data_type = values.get("data_type")
        name = values.get("name")
        object_id = values.get("object_id")

        if not out_dir.exists():
            msg = f"Output directory donot exist {out_dir}"
            raise ValueError(msg)

        conn_model = ServerConnection(
            username=OMERO_USERNAME,
            password=OMERO_PASSWORD,
            host=HOST,
            port=PORT,
        )
        conn = conn_model._authentication()
        conn.connect()
        data = conn.getObjects(data_type)
        ids = []
        names = []
        for d in data:
            ids.append(d.getId())
            names.append(d.getName())

        if name is not None and name not in names:
            msg = f"No such file is available {data_type}: name={name}"
            raise FileNotFoundError(msg)

        if object_id is not None and object_id not in ids:
            msg = f"No such file is available {data_type}: object_id={object_id}"
            raise FileNotFoundError(msg)
        conn.close()

        return values


class OmeroDwonload(CustomValidation):
    """Fetch data from an Omero Server.

    Args:
        data_type: The supported object types to be retreived.\
        Must be one of [project, dataset, screen, plate, well]
        name: Name of the object to be downloaded. Defaults to None.
        object_id: Identification of the object to be downloaded. Defaults to None.
        out_dir: The directory path for the outputs.

    Returns:
        microscopy image data.
    """

    data_type: str
    name: Optional[str] = None
    object_id: Optional[int] = None
    out_dir: Path

    def _create_output_directory(self, name: str) -> Path:
        """Create an output directory."""
        output = Path(self.out_dir).joinpath(name)

        if not output.exists():
            output.mkdir(exist_ok=True)

        return output

    def _write_ometif(self, image: np.ndarray, out_file: Path) -> None:
        """Utilizing BioWriter for writing numpy arrays."""
        with BioWriter(file_path=out_file) as bw:
            bw.X = image.shape[1]
            bw.Y = image.shape[0]
            bw.dtype = image.dtype
            bw[:, :, :, :, :] = image

    def _rename(self, x: str) -> str:
        """Rename a string."""
        return x.replace(".", "_")

    def _saveimage(  # noqa PLR0913
        self,
        image: np.ndarray,
        name: str,
        dir_name: Path,
        index: int,
        z: int,
        c: int,
        t: int,
    ) -> None:
        """Generating a single-plane image using BioWriter."""
        name = f"{name}_f{index}_z{z}_t{t}_c{c}.ome.tif"
        image_name = Path(dir_name, name)
        image = np.expand_dims(image, axis=(2, 3, 4))
        self._write_ometif(image, image_name)

    def get_data(self) -> None:  # noqa: PLR0912 PLR0915 C901
        """Retrieve data from the OMERO Server."""
        conn_model = ServerConnection(
            username=OMERO_USERNAME,
            password=OMERO_PASSWORD,
            host=HOST,
            port=PORT,
        )
        conn = conn_model._authentication()
        conn.connect()
        dc = DownloadControl()
        data = conn.getObjects(self.data_type)

        try:
            for d in data:
                if self.name is not None or self.object_id is not None:
                    if self.data_type == "project":  # noqa: SIM102
                        if d.getName() == self.name or d.getId() == self.object_id:
                            logger.info(
                                f"Downloading {self.data_type}: \
                                name={d.getName()} id={d.getId()}",
                            )
                            project_name = d.getName()
                            project_dir = self._create_output_directory(project_name)
                            for data in d.listChildren():
                                dataset_name = data.getName()
                                dataset_path = str(project_dir.joinpath(dataset_name))
                                dataset_dir = self._create_output_directory(
                                    dataset_path,
                                )
                                for image in data.listChildren():
                                    image_file = image.getFileset()
                                    if (
                                        image_file.__class__.__name__
                                        == "_FilesetWrapper"
                                    ):
                                        image_file = image.getFileset()
                                        dc.download_fileset(
                                            conn,
                                            image_file,
                                            dataset_dir,
                                        )

                                    if image_file is None:
                                        pixels = image.getPrimaryPixels().getPlane()
                                        for t, c, z in product(
                                            range(0, image.getSizeT()),
                                            range(0, image.getSizeC()),
                                            range(0, image.getSizeZ()),
                                        ):
                                            pixels = image.getPrimaryPixels().getPlane(
                                                theZ=z,
                                                theC=c,
                                                theT=t,
                                            )
                                            fimage = np.expand_dims(
                                                pixels,
                                                axis=(2, 3, 4),
                                            )
                                            outname = f"{image.getName()}.ome.tif"
                                            outfile = dataset_dir.joinpath(outname)
                                            self._write_ometif(fimage, outfile)

                    if self.data_type == "dataset":  # noqa: SIM102
                        if d.getName() == self.name or d.getId() == self.object_id:
                            logger.info(
                                f"Downloading {self.data_type}: \
                                name={d.getName()} id={d.getId()}",
                            )
                            dataset_name = d.getName()
                            dataset_dir = self._create_output_directory(dataset_name)
                            for image in d.listChildren():
                                image_file = image.getFileset()
                                if image_file.__class__.__name__ == "_FilesetWrapper":
                                    dc.download_fileset(conn, image_file, dataset_dir)
                                if image_file is None:
                                    pixels = image.getPrimaryPixels().getPlane()
                                    for t, c, z in product(
                                        range(0, image.getSizeT()),
                                        range(0, image.getSizeC()),
                                        range(0, image.getSizeZ()),
                                    ):
                                        pixels = image.getPrimaryPixels().getPlane(
                                            theZ=z,
                                            theC=c,
                                            theT=t,
                                        )
                                        fimage = np.expand_dims(pixels, axis=(2, 3, 4))
                                        outname = f"{image.getName()}.ome.tif"
                                        outfile = dataset_dir.joinpath(outname)
                                        self._write_ometif(fimage, outfile)

                    if self.data_type == "screen":  # noqa: SIM102
                        if d.getName() == self.name or d.getId() == self.object_id:
                            screen_name = d.getName()
                            screen_dir = self._create_output_directory(screen_name)
                            for plate in d.listChildren():
                                plate_name = plate.getName()
                                if plate_name == "MeasurementIndex.ColumbusIDX.xml":
                                    plate_name = self._rename(plate_name)
                                plate_name = screen_dir.joinpath(plate_name)
                                plate_dir = self._create_output_directory(plate_name)
                                for well in plate.listChildren():
                                    indicies = well.countWellSample()
                                    well_name = well.getWellPos()
                                    for index in range(0, indicies):
                                        pixels = well.getImage(index).getPrimaryPixels()
                                        for t, c, z in product(
                                            range(0, pixels.getSizeT()),
                                            range(0, pixels.getSizeC()),
                                            range(0, pixels.getSizeZ()),
                                        ):
                                            image = pixels.getPlane(
                                                theZ=z,
                                                theC=c,
                                                theT=t,
                                            )
                                            self._saveimage(
                                                image,
                                                well_name,
                                                plate_dir,
                                                index,
                                                z,
                                                c,
                                                t,
                                            )
                    if self.data_type == "plate":  # noqa: SIM102
                        if d.getName() == self.name or d.getId() == self.object_id:
                            plate_name = d.getName()
                            if plate_name == "MeasurementIndex.ColumbusIDX.xml":
                                plate_name = self._rename(plate_name)
                            plate_dir = self._create_output_directory(plate_name)
                            for well in d.listChildren():
                                indicies = well.countWellSample()
                                well_name = well.getWellPos()
                                for index in range(0, indicies):
                                    pixels = well.getImage(index).getPrimaryPixels()
                                    for t, c, z in product(
                                        range(0, pixels.getSizeT()),
                                        range(0, pixels.getSizeC()),
                                        range(0, pixels.getSizeZ()),
                                    ):
                                        image = pixels.getPlane(theZ=z, theC=c, theT=t)
                                        self._saveimage(
                                            image,
                                            well_name,
                                            plate_dir,
                                            index,
                                            z,
                                            c,
                                            t,
                                        )

                    if self.data_type == "well" and d.getId() == self.object_id:
                        well_pos = d.getWellPos()
                        well_id = d.getId()
                        well_name = f"well_{well_id}_{well_pos}"
                        well_dir = self._create_output_directory(well_name)
                        pixels = d.getImage().getPrimaryPixels()
                        for index, (t, c, z) in enumerate(
                            product(
                                range(0, pixels.getSizeT()),
                                range(0, pixels.getSizeC()),
                                range(0, pixels.getSizeZ()),
                            ),
                        ):
                            image = pixels.getPlane(theZ=z, theC=c, theT=t)
                            self._saveimage(image, well_name, well_dir, index, z, c, t)

            conn.close()

        except ValueError:
            logger.error("Invalid either object types, names or identifier")
