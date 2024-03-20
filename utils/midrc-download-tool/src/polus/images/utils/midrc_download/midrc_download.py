"""Midrc Download."""

import itertools
import logging
import os
import subprocess
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional
from typing import Union

import pandas as pd
import preadator
import requests
from gen3.auth import Gen3Auth
from polus.images.utils.midrc_download.utils import ENDPOINT
from polus.images.utils.midrc_download.utils import CustomValidation
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


cred = os.environ.get("MIDRC_API_KEY")

num_workers = max([cpu_count(), 2])

custom_hint = Union[str, list[str]]


class MIDRIC(CustomValidation):
    """A class designed for interacting with the Gen3 submission, query, and index APIs.

    Args:
        credentials: A Gen3Auth class instance.
        midrc_type: The node_id in the data model utilized in queries and API requests.
        project_id: The project code that this dataset belongs to the parent node.
        sex: A gender information.
        race: A race information.
        ethinicity: A racial or cultural background.
        age_at_index: The age of the study participant, at the time the imaging study.
        study_modality: The modalities of the imaging study
        body_part_examined: Body Part Examined
        loinc_contrast: The indicator if the image was completed w/wo contrast.
        loinc_method: The imaging modality associated with the assigned LOINC code.
        loinc_system: The body part examined associated with the assigned LOINC code.
        study_year: The year when imaging study was performed
        covid19_positive: An indicator if the patient has ever positive for COVID-19
        source_node: Image files & metadata related to several imaging series.
        data_format: The file format, physical medium, or dimensions of the resource.
        data_category: Image files and metadata related to several imaging series.
        data_type: The file format, physical medium, or dimensions of the resource
    """

    credentials: str
    midrc_type: str
    project_id: Optional[custom_hint] = None
    sex: Optional[custom_hint] = None
    race: Optional[custom_hint] = None
    ethnicity: Optional[custom_hint] = None
    age_at_index: Optional[custom_hint] = None
    study_modality: Optional[custom_hint] = None
    body_part_examined: Optional[custom_hint] = None
    loinc_contrast: Optional[custom_hint] = None
    loinc_method: Optional[custom_hint] = None
    loinc_system: Optional[custom_hint] = None
    study_year: Optional[custom_hint] = None
    covid19_positive: Optional[custom_hint] = None
    source_node: Optional[custom_hint] = None
    data_format: Optional[custom_hint] = None
    data_category: Optional[custom_hint] = None
    data_type: Optional[custom_hint] = None
    out_dir: Path

    def _authentication(self) -> tuple[str, str]:
        """Use credentials file for authentication."""
        return Gen3Auth(ENDPOINT, refresh_file=self.credentials)

    def transform_values(self, x: pd.DataFrame, feature: str) -> list:
        """Convert float feature values to strings.

        concatenate list values into a string separated by commas.
        """
        feat_lst = list(x[feature].values)
        feature_values = []
        string = ""
        for s in feat_lst:
            if isinstance(s, float):
                string = str(s)
            if isinstance(s, list):
                string = " ".join([str(item) for item in s])
            if isinstance(s, str):
                string = s
            feature_values.append(string)
        x[feature] = feature_values
        return x

    def get_query(
        self,
        x: dict[str, Union[str, int]],
    ) -> dict[str, dict[str, Union[str, int]]]:
        """Transform dictionary parameters into a GraphiQL query."""
        my_dict = {
            k: v
            for k, v in x.items()
            if k
            not in [
                "credentials",
                "midrc_type",
                "age_at_index",
                "study_year",
                "first",
                "offset",
                "out_dir",
            ]
        }
        fn: list = []
        for k, v in my_dict.items():
            if isinstance(v, (int, str)):
                fn_dict = {"=": {k: v}}
            if isinstance(v, list) and len(v) > 1:
                fn_dict = {"IN": {k: v}}
            fn.append(fn_dict)

        if x.get("age_at_index") is not None:
            if isinstance(x.get("age_at_index"), list):
                v1 = {
                    ">=": {
                        "age_at_index": int(x.get("age_at_index")[0]),  # type:ignore
                    },
                }
                v2 = {
                    "<=": {
                        "age_at_index": int(x.get("age_at_index")[1]),  # type:ignore
                    },
                }
                fn.append(v1)
                fn.append(v2)
            else:
                v1 = {">=": {"age_at_index": int(x.get("age_at_index"))}}  # type:ignore
                fn.append(v1)
        if x.get("study_year") is not None:
            if isinstance(x.get("study_year"), list):
                v1 = {">=": {"study_year": int(x.get("study_year")[0])}}  # type:ignore
                v2 = {"<=": {"study_year": int(x.get("study_year")[1])}}  # type:ignore
                fn.append(v1)
                fn.append(v2)
            else:
                v1 = {">=": {"study_year": int(x.get("study_year"))}}  # type: ignore
                fn.append(v1)

        return {"AND": fn}  # type: ignore

    def query_data(  # noqa: PLR0913
        self,
        midrc_type: str,
        fields: Union[list[str], None],
        filter_object: Optional[str | list[str]] = None,
        sort_fields: Optional[list[str]] = None,
        accessibility: Optional[str] = None,
        first: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        """Perform a data query against a MIDRC Data Commons.

        Args:
            midrc_type: MIDRC Data Common type.
            fields: Specify the list of fields to be returned.
            filter_object: Specify the filter to be applied.
            sort_fields: List of { field: sort method } objects.
            accessibility: Choose one option from: 'accessible', 'unaccessible', 'all'.
            first: Number of rows to return.
            offset: Starting position.

        Returns:
        List: A list of records represented as <record>.
        """
        url = f"{ENDPOINT}/guppy/download"

        if not accessibility:
            accessibility = "accessible"
        if not fields:
            fields = None
        try:
            body = {
                "type": midrc_type,
                "fields": fields,
                "accessibility": accessibility,
            }
            if filter_object:
                body["filter"] = filter_object
            if sort_fields:
                body["sort"] = sort_fields
            auth = self._authentication()
            response = requests.post(
                url,
                json=body,
                auth=auth,
                timeout=20,
            )
            status_code = 200
            if response.status_code != status_code:
                logger.error(f"Error: Unexpected response {response}")
            data = response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error: {e}")

        if offset:
            data = data[offset:]
        if first:
            data = data[:first]

        if len(data) > 0:
            logger.info("Successfully fetched records!!")
        else:
            msg = "Unable to fetch records!! Please query again"
            raise ValueError(msg)
        return data

    def download_data(self, data: list[dict]) -> None:
        """Execute a data download against a Data Commons.

        Args:
        data: MIDRC Data Common type to download from.
        """
        ## Simple loop to download all files and keep track of success and failures
        if len(data) > 0:
            df = pd.DataFrame(data)
            df.to_csv(self.out_dir.joinpath("metadata.csv"))

            object_ids = [i["object_id"] for i in data]
            object_ids = list(itertools.chain.from_iterable(object_ids))

            with preadator.ProcessManager(
                name="Downloading Midrc data",
                num_processes=num_workers,
                threads_per_process=2,
            ) as pm:
                count = 0
                for object_id in tqdm(object_ids, desc="Progress"):
                    count += 1
                    cmd = f"gen3 --auth {self.credentials} --endpoint data.midrc.org drs-pull object {object_id} --output-dir {self.out_dir}"  # noqa: E501
                    pm.submit_process(
                        subprocess.run,
                        cmd,
                        shell=True,  # noqa: S604
                        capture_output=True,
                    )

                pm.join_processes()
