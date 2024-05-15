"""Idr Download Package."""

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from itertools import chain
from itertools import product
from pathlib import Path
from typing import Any
from typing import Optional

import pandas as pd
import polus.images.utils.idr_download.utils as ut
import preadator
import requests
from omero.gateway import BlitzGateway
from pydantic import BaseModel as V2BaseModel
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

BOOL = False


class Connection(V2BaseModel):
    """Establishes a connection to an idr api server using BlitzGateway.

    Args:
        host: The IP address of the idr server.
        username:  The username for authentication.
        password:  The password for authentication.
        secure: Secure connection between client and server.

    Returns:
         BlitzGateway: A connection object to the IDR server.
    """

    def _authentication(self) -> BlitzGateway:
        """Connection to an idr server using BlitzGateway."""
        connection = BlitzGateway(
            host=ut.HOST,
            username=ut.USPW,
            passwd=ut.USPW,
            secure=True,
        )
        connection.connect()
        connection.c.enableKeepAlive(6)

        return connection


class Annotations(V2BaseModel):
    """Retrieve annotations for each well or image individually."""

    idx: int

    def well_annotations(self) -> pd.DataFrame:
        """Well Annotations."""
        well_url = f"{ut.WELL_URL}{self.idx}/query/?query=Well-{self.idx}"
        anno = requests.get(well_url, verify=BOOL, timeout=500).json()
        return pd.DataFrame(anno["data"]["rows"], columns=anno["data"]["columns"])

    def image_annotations(self) -> pd.DataFrame:
        """Image Annotations."""
        image_url = (
            f"{ut.BASE_URL}/webclient/api/annotations/?type=map&image={self.idx}"
        )
        keys = []
        values = []
        for a in requests.get(image_url, verify=BOOL, timeout=500).json()[
            "annotations"
        ]:
            for v in a["values"]:
                keys.append(v[0])
                values.append(v[1])
        return pd.DataFrame([values], columns=[keys])


class Collection(V2BaseModel):
    """Establishes a connection to an idr api server using BlitzGateway.

    Args:
        data_type: The supported object types to be retreived.
        out_dir:  Path to directory outputs.
        name:  Name of the object to be downloaded.
        object_id: Identifier of the object to be downloaded.
    """

    data_type: ut.DATATYPE
    out_dir: Path
    name: Optional[str] = None
    object_id: Optional[int] = None


class Project(Collection):
    """Obtain the dataset IDs linked to each project."""

    @property
    def get_data(self) -> tuple[list[list[Any]], Path]:
        """Retrieve dataset IDs linked to each project."""
        if self.name is not None:
            project_dict = ut._all_projects_ids()
            project_id = [i["id"] for i in project_dict if i["name"] == self.name]
            project_name = [
                i["projectName"] for i in project_dict if i["name"] == self.name
            ][0]

        if self.object_id is not None:
            project_dict = ut._all_projects_ids()
            project_id = [f["id"] for f in project_dict if f["id"] == self.object_id]
            project_name = [
                i["projectName"] for i in project_dict if i["id"] == project_id
            ][0]

        if len(project_id) == 0:
            msg = f"Please provide valid name or id of the {self.data_type}"
            raise ValueError(msg)

        dirpath = Path(self.out_dir, project_name)
        if not dirpath.exists():
            dirpath.mkdir(parents=True, exist_ok=False)

        conn = Connection()._authentication()
        conn.connect()

        dataset_list = []
        for _, pid in enumerate(project_id):
            project = conn.getObject(ut.DATATYPE.PROJECT, pid)
            datasets = [dataset.getId() for dataset in project.listChildren()]
            dataset_list.append(datasets)

        dataset_list = list(chain.from_iterable(dataset_list))
        conn.close()

        return dataset_list, dirpath


class Dataset(Collection):
    """Download the IDR dataset object."""

    @property
    def get_data(self) -> None:
        """Write all the images of the IDR dataset object."""
        conn = Connection()._authentication()
        conn.connect()
        dataset = conn.getObject(ut.DATATYPE.DATASET, self.object_id)
        if dataset.__class__.__name__ == ut.DATATYPE_MAPPING[self.data_type]:
            dataset_name = dataset.getName()
            dataset_dir = self.out_dir.joinpath(dataset_name)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            anno_dir = self.out_dir.joinpath(Path("metadata", dataset_name))
            anno_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame()
            with ThreadPoolExecutor(max_workers=15) as executor:
                for image in dataset.listChildren():
                    image_id = image.getId()
                    image_name = image.getName().split(".")[0]
                    df_anno = Annotations(idx=image_id).image_annotations()
                    df_anno.to_csv(
                        Path(anno_dir, f"{dataset_name}_{image_name}.csv"),
                        index=False,
                    )
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

                        imagename = image.getName().split(".")[0]
                        executor.submit(
                            ut.saveimage(pixels, imagename, dataset_dir, z, c, t),
                        )

        conn.close()
        if dataset.__class__.__name__ == "NoneType":
            msg = f"Please provide valid name or id of the {self.data_type}"
            raise ValueError(msg)


class Screen(Collection):
    """Obtain the plate IDs and names linked to each screen."""

    @property
    def plates(self) -> tuple[list[dict], Path]:
        """Retrieve the plate IDs and names linked to each screen."""
        screen_dict = ut._all_screen_ids()

        if self.name is not None:
            screen = [
                {"id": f["id"], "name": f["screenName"]}
                for f in screen_dict
                if f["name"] == self.name
            ][0]

        if self.object_id is not None:
            screen = [
                {"id": f["id"], "name": f["screenName"]}
                for f in screen_dict
                if f["id"] == self.object_id
            ][0]

        if len(screen) == 0:
            msg = f"Please provide valid name or id of the {self.data_type}"
            raise ValueError(msg)

        dirpath = Path(self.out_dir, screen["name"])
        if not dirpath.exists():
            dirpath.mkdir(parents=True, exist_ok=False)

        screen_id = screen["id"]

        plates_url = f"{ut.BASE_URL}/webclient/api/plates/?id={screen_id}"
        return [
            {"id": p["id"], "name": p["name"]}
            for p in requests.get(plates_url, verify=BOOL, timeout=500).json()["plates"]
        ], dirpath


class Plate(Collection):
    """Download the IDR plate object."""

    @property
    def get_data(self) -> None:
        """Save all images from the plate object."""
        conn = Connection()._authentication()
        conn.connect()
        plate = conn.getObject(ut.DATATYPE.PLATE, self.object_id)
        if plate.__class__.__name__ == ut.DATATYPE_MAPPING[self.data_type]:
            plate_name = plate.getName()
            plate_dir = self.out_dir.joinpath(plate_name)
            anno_dir = self.out_dir.joinpath("metadata")
            plate_dir.mkdir(parents=True, exist_ok=True)
            anno_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame()
            with ThreadPoolExecutor(max_workers=ut.NUM_THREADS) as executor:
                threads = []
                for _, well in enumerate(plate.listChildren()):
                    data_type = "well"
                    if well.__class__.__name__ == ut.DATATYPE_MAPPING[data_type]:
                        indicies = well.countWellSample()
                        well_name = well.getWellPos()
                        well_id = well.getId()
                        df_anno = Annotations(idx=well_id).well_annotations()
                        data = [df, df_anno]
                        df = pd.concat(data, ignore_index=True, sort=False)
                        df.to_csv(Path(anno_dir, f"{plate_name}.csv"), index=False)
                        for index in range(0, indicies):
                            pixels = well.getImage(index).getPrimaryPixels()
                            for t, c, z in product(
                                range(0, pixels.getSizeT()),
                                range(0, pixels.getSizeC()),
                                range(0, pixels.getSizeZ()),
                            ):
                                image = pixels.getPlane(theZ=z, theC=c, theT=t)
                                threads.append(
                                    executor.submit(
                                        ut.saveimage(
                                            image,
                                            well_name,
                                            plate_dir,
                                            z,
                                            c,
                                            t,
                                            index,
                                        ),
                                    ),
                                )

            for future in tqdm(
                as_completed(threads),
                total=len(threads),
                desc="Fetching wells",
            ):
                plate = future.result()

        conn.close()


class Well(Collection):
    """Download the IDR well object."""

    @property
    def get_data(self) -> None:
        """Save all images from the well object."""
        conn = Connection()._authentication()
        conn.connect()
        well = conn.getObject(ut.DATATYPE.WELL, self.object_id)
        if well.__class__.__name__ == ut.DATATYPE_MAPPING[self.data_type]:
            indicies = well.countWellSample()
            well_name = well.getWellPos()
            well_id = well.getId()
            df_anno = Annotations(idx=well_id).well_annotations()
            well_dir = self.out_dir.joinpath(well_name)
            well_dir.mkdir(parents=True, exist_ok=True)
            anno_dir = self.out_dir.joinpath("metadata")
            anno_dir.mkdir(parents=True, exist_ok=True)
            df_anno.to_csv(Path(anno_dir, f"{well_id}_{well_name}.csv"), index=False)
            with ThreadPoolExecutor(max_workers=ut.NUM_THREADS) as executor:
                for index in range(0, indicies):
                    pixels = well.getImage(index).getPrimaryPixels()
                    for t, c, z in product(
                        range(0, pixels.getSizeT()),
                        range(0, pixels.getSizeC()),
                        range(0, pixels.getSizeZ()),
                    ):
                        image = pixels.getPlane(theZ=z, theC=c, theT=t)
                        executor.submit(
                            ut.saveimage(image, well_name, well_dir, z, c, t, index),
                        )
        conn.close()


class IdrDownload(Collection):
    """Download the IDR objects."""

    @property
    def get_data(self) -> None:  # noqa : C901
        """Save all images from the IDR objects."""
        if self.data_type == ut.DATATYPE.SCREEN:
            if self.name is not None:
                sc, dirpath = Screen(
                    data_type=self.data_type,
                    name=self.name,
                    out_dir=self.out_dir,
                ).plates
                logger.info(f"Downloading {self.data_type}: name={self.name}")
            if self.object_id is not None:
                sc, dirpath = Screen(
                    data_type=self.data_type,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).plates
                logger.info(f"Downloading {self.data_type}: id={self.object_id}")
            if self.name is not None and self.object_id is not None:
                sc, dirpath = Screen(
                    data_type=self.data_type,
                    name=self.name,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).plates
                logger.info(
                    f"Download {self.data_type}:name={self.name},id={self.object_id}",
                )
            if self.name is None and self.object_id is None:
                msg = f"Both {self.data_type} name & {self.data_type} id is missing"
                raise ValueError(msg)
            plate_list = sc
            with preadator.ProcessManager(
                name="Idr download",
                num_processes=4,
                threads_per_process=2,
            ) as executor:
                threads = []
                for _, pl in enumerate(plate_list):
                    plate_id = pl["id"]
                    threads.append(
                        executor.submit(
                            Plate(
                                data_type=ut.DATATYPE.PLATE,
                                object_id=plate_id,
                                out_dir=dirpath,
                            ).get_data,
                        ),
                    )

                for f in tqdm(
                    as_completed(threads),
                    total=len(threads),
                    mininterval=5,
                    desc=f"download plate {plate_id}",
                    initial=0,
                    unit_scale=True,
                    colour="cyan",
                ):
                    f.result()

        if self.data_type == ut.DATATYPE.PLATE:
            if self.name is not None:
                plate_id = [
                    pl["id"] for pl in ut._all_plates_ids() if pl["name"] == self.name
                ][0]
                Plate(  # noqa:B018
                    data_type=self.data_type,
                    object_id=plate_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(f"Downloading {self.data_type}: name={self.name}")
            if self.object_id is not None:
                Plate(  # noqa:B018
                    data_type=self.data_type,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(f"Downloading {self.data_type}: id={self.object_id}")
            if self.name is not None and self.object_id is not None:
                Plate(  # noqa:B018
                    data_type=self.data_type,
                    name=self.name,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(
                    f"Download {self.data_type}:name={self.name},id={self.object_id}",
                )
            if self.name is None and self.object_id is None:
                msg = f"Both {self.data_type} name & {self.data_type} id are missing"
                raise ValueError(msg)

        if self.data_type == ut.DATATYPE.WELL:
            if self.object_id is None:
                msg = f"Please provide objectID of {self.data_type}"
                raise ValueError(msg)

            Well(  # noqa:B018
                data_type=self.data_type,
                object_id=self.object_id,
                out_dir=self.out_dir,
            ).get_data

        if self.data_type == ut.DATATYPE.PROJECT:
            if self.object_id is None and self.name is None:
                msg = f"Both {self.data_type} name & {self.data_type} id are missing"
                raise ValueError(msg)
            if self.name is not None:
                dataset_list, dirpath = Project(
                    data_type=self.data_type,
                    name=self.name,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(f"Downloading {self.data_type}: name={self.name}")
            if self.object_id is not None:
                dataset_list, dirpath = Project(
                    data_type=self.data_type,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(f"Downloading {self.data_type}: id={self.object_id}")
            if self.name is not None and self.object_id is not None:
                dataset_list, dirpath = Project(
                    data_type=self.data_type,
                    name=self.name,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(
                    f"Download {self.data_type}:name={self.name},id={self.object_id}",
                )

            for d in dataset_list:
                Dataset(  # noqa:B018 # type:ignore
                    data_type=ut.DATATYPE.DATASET,
                    object_id=d,
                    out_dir=dirpath,
                ).get_data

        if self.data_type == ut.DATATYPE.DATASET:
            if self.object_id is None and self.name is None:
                msg = f"Both {self.data_type} name & {self.data_type} id are missing"
                raise ValueError(msg)
            if self.name is not None:
                dataset_ids = ut._all_datasets_ids()
                data_id = [d["id"] for d in dataset_ids if d["name"] == self.name][0]
                Dataset(  # noqa:B018
                    data_type=self.data_type,
                    object_id=data_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(f"Downloading {self.data_type}: name={self.name}")
            if self.object_id is not None:
                Dataset(  # noqa:B018
                    data_type=self.data_type,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(f"Downloading {self.data_type}: id={self.object_id}")
            if self.name is not None and self.object_id is not None:
                Dataset(  # noqa:B018
                    data_type=self.data_type,
                    name=self.name,
                    object_id=self.object_id,
                    out_dir=self.out_dir,
                ).get_data
                logger.info(
                    f"Download {self.data_type}:name={self.name},id={self.object_id}",
                )
