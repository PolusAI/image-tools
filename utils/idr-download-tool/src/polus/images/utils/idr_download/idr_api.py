"""Idr Download Package."""

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
import requests

BASE_URL = "https://idr.openmicroscopy.org"
HOST = "ws://idr.openmicroscopy.org/omero-ws"
USPW = "public"
SCREEN_URL = f"{BASE_URL}/api/v0/m/screens/"

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


# def generate_preview(
#     path: Path,
# ) -> None:
#     """Generate preview of the plugin outputs."""
#     prev_file = list(
#         Path().cwd().parents[4].joinpath("examples").rglob(f"*{POLUS_EXT}"),
#     )[0]

#     shutil.copy(prev_file, path)


class DATATYPE(str, Enum):
    """Objects types."""

    PROJECT = "project"
    DATASET = "dataset"
    SCREEN = "screen"
    PLATE = "plate"
    WELL = "well"
    Default = "dataset"

class Connection(V2BaseModel):
    """Establishes a connection to an idr api server using BlitzGateway.

    Args:
        username:  The username for authentication.
        password:  The password for authentication
        host: The IP address of the idr server.
        port: Port used to establish a connection between client and server.

    Returns:
         BlitzGateway: A connection object to the IDR server.
    """

    def _authentication(self) -> BlitzGateway:
        """Connection to an idr server using BlitzGateway."""
        return BlitzGateway(
            host=HOST,
            username=USPW,
            passwd=USPW,
            secure=True,
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

        conn_model = Connection(
            host=HOST,
            username=USPW,
            password=USPW,
            secure=True
        )
        conn = conn_model._authentication()
        conn.connect()

        response = requests.get(SCREEN_URL).json()
        screen_names = []
        screen_ids = []
        for r in response['data']:
            screen_names.append(r['Name'].split("-")[0])
            screen_ids.append(r["@id"])
    
        if data_type == "screen":    
            if not name in screen_names:
                msg = f"Name of {data_type} is incorrect"
                raise ValueError(msg)
            if object_id is not None:
                if not object_id in screen_ids:
                    msg = f"Object id of {data_type} is incorrect"
                    raise ValueError(msg)
                
        if data_type == "plate":
            plate_dict = []
            for i in screen_ids:
                PLATES_URL = f"{BASE_URL}/webclient/api/plates/?id={i}"
                for p in requests.get(PLATES_URL).json()['plates']:
                    new_dict = {'id': p['id'], 'name': p['name']}
                    plate_dict.append(new_dict)

            if object_id is not None:
                plate = conn.getObject("plate", object_id)
                if not plate.__class__.__name__ == "_PlateWrapper":
                    msg = f"Object id of {data_type} is incorrect"
                    raise ValueError(msg)
            
            if name is not None:
                for p in plate_dict:
                    if name == p['name']:
                        plate = conn.getObject("plate", p['id'])
                        plate_name = plate.getName()
                        if not name == plate_name:
                            msg = f"Name of {data_type} is incorrect"
                            raise ValueError(msg)

        conn.close()


        
        return values
    


class IdrDwonload(CustomValidation):
    data_type: str
    name: Optional[str] = None
    object_id: Optional[int] = None
    out_dir: Path







        #         URL = "https://idr.openmicroscopy.org/api/v0/m/screens/"
        # response = requests.get(URL).json()

        # for r in response['data']:
        #     print(r['Name'], r["@id"])
        # data = conn.getObjects(data_type)
        # ids = []
        # names = []
        # for d in data:
        #     ids.append(d.getId())
        #     names.append(d.getName())

        # if name is not None and name not in names:
        #     msg = f"No such file is available {data_type}: name={name}"
        #     raise FileNotFoundError(msg)

        # if object_id is not None and object_id not in ids:
        #     msg = f"No such file is available {data_type}: object_id={object_id}"
        #     raise FileNotFoundError(msg)
        # conn.close()

        # return values


