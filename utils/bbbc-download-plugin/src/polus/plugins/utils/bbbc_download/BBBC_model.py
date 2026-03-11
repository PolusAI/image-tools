"""BBBC dataset models and download logic."""  # noqa: N999
from __future__ import annotations

import contextlib
import logging
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Generic
from typing import TypeVar
from typing import cast
from zipfile import ZipFile

import bs4
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from bfio import BioWriter
from polus.plugins.utils.bbbc_download.download import download
from polus.plugins.utils.bbbc_download.download import get_url
from polus.plugins.utils.bbbc_download.download import remove_macosx
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator
from skimage import io
from tqdm import tqdm

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


class _ClassProperty(Generic[_T]):
    """Descriptor: MyClass.attr returns the result of a classmethod when accessed."""

    def __init__(self, f: Callable[..., _T]) -> None:
        self.f = f

    def __get__(self, obj: object, owner: type) -> _T:
        if owner is None:
            msg = "ClassProperty accessed without owner"
            raise RuntimeError(msg)
        return self.f(owner)


BASE_URL = "https://bbbc.broadinstitute.org/"
tables = pd.read_html(BASE_URL + "image_sets")[:3]
root = Path("/BBBC").absolute()


exception_sets = [
    "BBBC019",
    "BBBC029",
    "BBBC041",
    "BBBC042",
    "BBBC046",
    "BBBC054",
]

# Cleaning up column names
tables[0] = tables[0].rename(
    columns={
        tables[0].columns[3]: "Fields per sample",
        tables[0].columns[5]: "Total Images",
        tables[0].columns[6]: "Ground truth",
    },
)


class Metadata(BaseModel):
    """Class that contains information about a dataset's metadata."""

    path: Path
    name: str

    @model_validator(mode="after")
    def valid_data(self) -> Metadata:  # noqa: D102
        if not self.path.exists():
            msg = "No metadata"
            raise ValueError(msg)
        return self

    @property
    def size(self) -> int:
        """Returns the size of the dataset's metadata in bytes."""
        raw_path = root.joinpath(self.name, "raw/Metadata")
        standard_path = root.joinpath(self.name, "standard/Metadata")
        raw_sum = sum(os.path.getsize(file) for file in raw_path.rglob("*"))
        standard_sum = sum(os.path.getsize(file) for file in standard_path.rglob("*"))

        return raw_sum + standard_sum


class GroundTruth(BaseModel):
    """Class that contains information about a dataset's ground truth."""

    path: Path
    name: str

    @model_validator(mode="after")
    def valid_data(self) -> GroundTruth:  # noqa: D102
        if not self.path.exists():
            msg = "No ground truth"
            raise ValueError(msg)
        return self

    @property
    def size(self) -> int:
        """Returns the size of the dataset's ground truth in bytes."""
        raw_path = root.joinpath(self.name, "raw/Ground_Truth")
        standard_path = root.joinpath(self.name, "standard/Ground_Truth")
        raw_sum = sum(os.path.getsize(file) for file in raw_path.rglob("*"))
        standard_sum = sum(os.path.getsize(file) for file in standard_path.rglob("*"))

        return raw_sum + standard_sum


class Images(BaseModel):
    """Class that contains information about a dataset's images."""

    path: Path
    name: str

    @model_validator(mode="after")
    def valid_data(self) -> Images:  # noqa: D102
        if not self.path.exists():
            msg = "No images"
            raise ValueError(msg)
        return self

    @property
    def size(self) -> int:
        """Returns the size of the dataset's images in bytes."""
        raw_path = root.joinpath(self.name, "raw/Images")
        standard_path = root.joinpath(self.name, "standard/Images")
        raw_sum = sum(os.path.getsize(file) for file in raw_path.rglob("*"))
        standard_sum = sum(os.path.getsize(file) for file in standard_path.rglob("*"))

        return raw_sum + standard_sum


class BBBCDataset(BaseModel):
    """Class that models a BBBC dataset.

    Attributes:
        name: The name of the dataset.
        images: Images object for the dataset's images.
        ground_truth: GroundTruth object for the dataset's ground truth.
        metadata: Metadata object for the dataset's metadata.
    """

    name: str
    images: Images | None = None
    ground_truth: GroundTruth | None = None
    metadata: Metadata | None = None
    output_path: Path | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, v: str) -> str:
        """Validates the name of the dataset.

        Args:
            v: The name of the dataset to be downloaded.

        Returns:
            The name provided if validation is successful.
        """
        combined = cast(pd.DataFrame, BBBC.combined_table)
        if v not in list(combined["Accession"]):
            msg = (
                f"{v} is an invalid dataset name. "
                "Valid names belong to an existing BBBC dataset."
            )
            raise ValueError(msg)

        return v

    @classmethod
    def create_dataset(cls, name: str) -> BBBCDataset | None:
        """Creates a dataset.

        Args:
            name: The name of the dataset to be created.

        Returns:
            A new instance of a Dataset object or None if the validation fails.
        """
        try:
            if name in exception_sets:
                dataset_class = globals()[name]
                return dataset_class(name=name)
            return BBBCDataset(name=name)
        except ValueError as e:
            logger.info(f"{e}")

            return None

    @property
    def info(self) -> dict[str, str | np.int64]:
        """Provide dataset info (description, total images, etc.).

        Returns:
            A dictionary that contains information about the dataset.
        """
        table = cast(pd.DataFrame, BBBC.combined_table)
        row = table.loc[table["Accession"] == self.name]

        return {
            "Description": row["Description"].values[0],
            "Mode": row["Mode"].values[0],
            "Fields per sample": row["Fields per sample"].values[0],
            "Total Fields": row["Total Fields"].values[0],
            "Total Images": row["Total Images"].values[0],
            "Ground truth types": self._ground_truth_types(),
        }

    @property
    def size(self) -> int:
        """Returns the size of the dataset in bytes."""
        if self.output_path is None:
            return 0
        dataset_path = self.output_path.joinpath("BBBC", self.name)

        return sum(os.path.getsize(file) for file in dataset_path.rglob("*"))

    def _ground_truth_types(self) -> list[str]:
        """Provides the types of ground truth used by the dataset.

        Returns:
            A list of strings where each string is a type of ground truth.
        """
        res = requests.get(
            "https://bbbc.broadinstitute.org/image_sets",
            timeout=30,
        )
        soup = bs4.BeautifulSoup(res.content, "html.parser")
        types = []

        for t in soup.find_all("table")[:3]:
            for row in t.find_all("tr"):
                cols = row.find_all("td")

                if len(cols) > 0 and cols[0].text == self.name:
                    for link in cols[6].find_all("a"):
                        types.append(link.attrs["href"].split("#")[-1])

                    return types
        return []

    def _init_data(self, download_path: Path) -> None:
        """Initialize images, ground_truth, and metadata from download_path."""
        download_path = download_path.joinpath("BBBC")

        images_path = download_path.joinpath(self.name, "raw/Images")
        truth_path = download_path.joinpath(self.name, "raw/Ground_Truth")
        meta_path = download_path.joinpath(self.name, "raw/Metadata")

        with contextlib.suppress(ValueError):
            self.images = Images(path=images_path, name=self.name)

        with contextlib.suppress(ValueError):
            self.ground_truth = GroundTruth(path=truth_path, name=self.name)

        with contextlib.suppress(ValueError):
            self.metadata = Metadata(path=meta_path, name=self.name)

        if self.images is None:
            logger.info(f"{self.name} has no images")

        if self.ground_truth is None and self.metadata is None:
            logger.info(f"{self.name} has no ground truth or metadata")

    def raw(self, download_path: Path) -> None:
        """Download the dataset's raw data."""
        self.output_path = download_path

        download(self.name, download_path)
        self._init_data(download_path)

    def standard(self, extension: str) -> None:
        """Standardize the dataset's raw data.

        Args:
            extension: Standard image extension: ".ome.tif" or ".ome.zarr".
        """
        if extension not in [".ome.tif", ".ome.zarr"]:
            logger.info(
                "ERROR: %s is an invalid extension for standardization. "
                "Must be .ome.tif or .ome.zarr.",
                extension,
            )
            return

        if self.images is None:
            logger.info(
                "ERROR: Images for %s have not been downloaded so they cannot be "
                "standardized.",
                self.name,
            )
            return

        standard_folder = Path(root, self.name, "standard")
        arrow_file = Path("arrow", self.name + ".arrow")
        arrow_table = pq.read_table(arrow_file)
        df = arrow_table.to_pandas()

        if not standard_folder.exists():
            standard_folder.mkdir(parents=True, exist_ok=True)

        image_ndim_2d = 2
        for _, row in df.iterrows():
            func = globals()[self.name + "_mapping"]
            out_file = func(row, extension)
            raw_image = io.imread(row["Path"])
            num_channels = (
                1 if len(raw_image.shape) == image_ndim_2d else raw_image.shape[2]
            )

            if row["Image Type"] == "Intensity":
                sub_folder = "Images"
            elif row["Image Type"] == "Ground Truth":
                sub_folder = "Ground_Truth"
            elif row["Image Type"] == "Metadata":
                sub_folder = "Metadata"
            else:
                logger.info("ERROR: Invalid value for attribute Image Type")
                return

            save_path = standard_folder.joinpath(sub_folder)

            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)

            with BioWriter(save_path.joinpath(out_file)) as bw:
                bw.X, bw.Y, bw.Z, bw.C = (
                    raw_image.shape[1],
                    raw_image.shape[0],
                    num_channels,
                    1,
                )
                bw.dtype = raw_image.dtype
                bw[:] = raw_image

        logger.info(f"Finished standardizing {self.name}")

        return


class BBBC019(BBBCDataset):
    """BBBC019 dataset with custom raw download/organization."""

    def raw(self, download_path: Path) -> None:
        """Download and organize BBBC019 raw data."""
        download(self.name, download_path)
        self.output_path = download_path
        save_location = download_path.joinpath("BBBC")

        # Separate images from ground truth
        save_location = save_location.joinpath("BBBC019")
        images_folder = save_location.joinpath("raw/Images")
        truth_folder = save_location.joinpath("raw/Ground_Truth")
        for folder in [
            x
            for x in images_folder.iterdir()
            if x.name not in [".DS_Store", "__MACOSX"]
        ]:
            for obj in [
                x
                for x in folder.iterdir()
                if x.name not in ["images", "measures.mat", "desktop.ini", ".DS_Store"]
            ]:
                src = images_folder.joinpath(folder.name, obj.name)
                dst = truth_folder.joinpath(folder.name, obj.name)

                if dst.exists():
                    try:
                        shutil.rmtree(src)
                    except NotADirectoryError as e:
                        logger.info(f"{e}")
                else:
                    shutil.move(src, dst)

        self._init_data(download_path)


class BBBC029(BBBCDataset):
    """BBBC029 dataset with custom raw download/organization."""

    def raw(self, download_path: Path) -> None:
        """Download and organize BBBC029 raw data."""
        logger.info("Started downloading BBBC029")
        self.output_path = download_path
        save_location = download_path.joinpath("BBBC")

        save_location = save_location.joinpath("BBBC029", "raw")

        if not save_location.exists():
            save_location.mkdir(parents=True, exist_ok=True)

        file_path = save_location.joinpath("Images")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC029/images.zip",
            file_path,
            "BBBC029",
        )

        file_path = save_location.joinpath("Ground_Truth")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC029/ground_truth.zip",
            file_path,
            "BBBC029",
        )

        logger.info("BBBC029 has finished downloading")
        images_folder = save_location.joinpath("Images")
        truth_folder = save_location.joinpath("Ground_Truth")
        remove_macosx("BBBC029", images_folder)
        remove_macosx("BBBC029", truth_folder)
        source_directory = images_folder.joinpath("images")
        for source_file in source_directory.glob("*"):
            destination_file = images_folder / source_file.name
            shutil.move(source_file, destination_file)
        shutil.rmtree(source_directory)

        source_directory = truth_folder.joinpath("ground_truth")
        for source_file in source_directory.glob("*"):
            destination_file = truth_folder / source_file.name
            shutil.move(source_file, destination_file)
        shutil.rmtree(source_directory)

        self._init_data(download_path)


class BBBC041(BBBCDataset):
    """BBBC041 dataset with custom raw download/organization."""

    def raw(self, download_path: Path) -> None:
        """Download and organize BBBC041 raw data."""
        download(self.name, download_path)
        self.output_path = download_path
        save_location = download_path.joinpath("BBBC")

        # Separate images from ground truth
        save_location = save_location.joinpath("BBBC041")
        file_names = ["test.json", "training.json"]

        if not save_location.joinpath("raw/Ground_Truth").exists():
            save_location.joinpath("raw/Ground_Truth").mkdir(
                parents=True,
                exist_ok=True,
            )

        for file in file_names:
            src = save_location.joinpath("raw/Images/malaria", file)
            dst = save_location.joinpath("raw/Ground_Truth")

            if dst.joinpath(file).exists():
                src.unlink(missing_ok=True)
            else:
                shutil.move(src, dst)

        self._init_data(download_path)


class BBBC042(BBBCDataset):
    """BBBC042 dataset with custom raw download/organization."""

    def raw(self, download_path: Path) -> None:
        """Download and organize BBBC042 raw data."""
        logger.info("Started downloading BBBC042")
        self.output_path = download_path
        save_location = download_path.joinpath("BBBC")

        save_location = save_location.joinpath("BBBC042", "raw")

        if not save_location.exists():
            save_location.mkdir(parents=True, exist_ok=True)

        file_path = save_location.joinpath("Images")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC042/images.zip",
            file_path,
            "BBBC042",
        )

        file_path = save_location.joinpath("Ground_Truth")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC042/positions.zip",
            file_path,
            "BBBC042",
        )

        logger.info("BBBC042 has finished downloading")
        images_folder = save_location.joinpath("Images")
        truth_folder = save_location.joinpath("Ground_Truth")
        remove_macosx("BBBC042", images_folder)
        remove_macosx("BBBC042", truth_folder)

        self._init_data(download_path)


class BBBC046(BBBCDataset):
    """BBBC046 dataset with custom raw download/organization."""

    def raw(self, download_path: Path) -> None:
        """Download and organize BBBC046 raw data."""
        download(self.name, download_path)
        self.output_path = download_path
        save_location = download_path.joinpath("BBBC")

        # Separate images from ground truth
        try:
            save_location = save_location.joinpath(self.name)
            images_folder = save_location.joinpath("raw/Images")
            truth_folder = save_location.joinpath("raw/Ground_Truth")

            # Extract these files because they do not extract automatically
            for file in [
                "OE-ID350-AR-1.zip",
                "OE-ID350-AR-2.zip",
                "OE-ID350-AR-4.zip",
                "OE-ID350-AR-8.zip",
            ]:
                with ZipFile(images_folder.joinpath(file), "r") as zfile:
                    zfile.extractall(images_folder)

                images_folder.joinpath(file).unlink(missing_ok=True)

            if not truth_folder.exists():
                truth_folder.mkdir(parents=True, exist_ok=True)

            # Iterate over folders in the images folder
            for folder in images_folder.iterdir():
                if not truth_folder.joinpath(folder.name).exists():
                    truth_folder.joinpath(folder.name).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                # Move ground truth data to Ground Truth folder
                for obj in folder.iterdir():
                    if obj.name.endswith((".txt", ".tif")):
                        src = obj
                        dst = truth_folder.joinpath(folder.name, obj.name)

                        if dst.exists():
                            src.unlink(missing_ok=True)
                        else:
                            shutil.move(src, dst)

            self._init_data(download_path)
        except (OSError, ValueError) as e:
            logger.info(
                "BBBC046 downloaded successfully but an error occurred when "
                "organizing raw data.",
            )
            logger.info("ERROR: %s", e)


class BBBC054(BBBCDataset):
    """BBBC054 dataset with custom raw download/organization."""

    def raw(self, download_path: Path) -> None:
        """Download and organize BBBC054 raw data."""
        download(self.name, download_path)
        self.output_path = download_path
        save_location = download_path.joinpath("BBBC")

        # Separate images from ground truth
        save_location = save_location.joinpath(self.name)
        src = save_location.joinpath("raw/Images", "Replicate1annotation.csv")
        dst = save_location.joinpath("raw/Ground_Truth", "Replicate1annotation.csv")

        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            src.unlink(missing_ok=True)
        else:
            shutil.move(src, dst)

        self._init_data(download_path)


class IDAndSegmentation:
    """Class that models the Identification and segmentation table on https://bbbc.broadinstitute.org/image_sets.

    Attributes:
        name: The name of the table as seen on the BBBC image set webpage
        table: The Identification and segmentation table as a pandas DataFrame
    """

    name: str = "Identification and segmentation"
    table: pd.DataFrame = tables[0]

    @_ClassProperty
    def datasets(cls) -> list[BBBCDataset]:  # noqa: N805
        """Returns a list of all datasets in the table.

        Returns:
            A list containing a Dataset object for each dataset in the table.
        """
        return [
            d
            for name in cls.table["Accession"]
            if (d := BBBCDataset.create_dataset(name)) is not None
        ]

    @classmethod
    def raw(cls, download_path: Path) -> None:
        """Downloads raw data for every dataset in this table."""
        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in IDAndSegmentation.datasets:
                threads.append(executor.submit(dataset.raw, download_path))

            for f in tqdm(
                as_completed(threads),
                desc="Downloading data",
                total=len(threads),
            ):
                f.result()


class PhenotypeClassification:
    """Class that models the Phenotype classification table on https://bbbc.broadinstitute.org/image_sets.

    Attributes:
        name: The name of the table as seen on the BBBC image set webpage
        table: The Phenotype classification table as a pandas DataFrame
    """

    name: str = "Phenotype classification"
    table: pd.DataFrame = tables[1]

    @_ClassProperty
    def datasets(cls) -> list[BBBCDataset]:  # noqa: N805
        """Returns a list of all datasets in the table.

        Returns:
            A list containing a Dataset object for each dataset in the table.
        """
        return [
            d
            for name in cls.table["Accession"]
            if (d := BBBCDataset.create_dataset(name)) is not None
        ]

    @classmethod
    def raw(cls, download_path: Path) -> None:
        """Downloads raw data for every dataset in this table."""
        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in PhenotypeClassification.datasets:
                threads.append(executor.submit(dataset.raw, download_path))

            for f in tqdm(
                as_completed(threads),
                desc="Downloading data",
                total=len(threads),
            ):
                f.result()


class ImageBasedProfiling:
    """Class that models the Image-based Profiling table on https://bbbc.broadinstitute.org/image_sets.

    Attributes:
        name: The name of the table as seen on the BBBC image set webpage
        table: The Image-based Profiling table as a pandas DataFrame
    """

    name: str = "Image-based Profiling"
    table: pd.DataFrame = tables[2]

    @_ClassProperty
    def datasets(cls) -> list[BBBCDataset]:  # noqa: N805
        """Returns a list of all datasets in the table.

        Returns:
            A list containing a Dataset object for each dataset in the table.
        """
        return [
            d
            for name in cls.table["Accession"]
            if (d := BBBCDataset.create_dataset(name)) is not None
        ]

    @classmethod
    def raw(cls, download_path: Path) -> None:
        """Downloads raw data for every dataset in this table."""
        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in ImageBasedProfiling.datasets:
                threads.append(executor.submit(dataset.raw, download_path))

            for f in tqdm(
                as_completed(threads),
                desc="Downloading data",
                total=len(threads),
            ):
                f.result()


class BBBC:
    """Class that models the Broad Bioimage Benchmark Collection (BBBC).

    BBBC has tables that contain datasets. Datasets are separated into tables
    based on how they can be used. Each dataset has images and ground truth.
    Read more about BBBC here: https://bbbc.broadinstitute.org.
    """

    @_ClassProperty
    def datasets(cls) -> list[BBBCDataset]:  # noqa: N805
        """Returns a list of all datasets in BBBC.

        Returns:
            A list containing a Dataset object for each dataset in BBBC.
        """
        table = cast(pd.DataFrame, BBBC.combined_table)
        return [
            d
            for name in table["Accession"]
            if (d := BBBCDataset.create_dataset(name)) is not None
        ]

    @_ClassProperty
    def combined_table(cls) -> pd.DataFrame:  # noqa: N805
        """Combine image_sets tables into a single table.

        Returns:
            A pandas DataFrame representation of the combined table.
        """
        # Combine each table into one table
        return (
            pd.concat(tables)
            .drop(columns=["Ground truth"])
            .drop_duplicates("Accession")
        )

    @classmethod
    def raw(cls, download_path: Path) -> None:
        """Downloads raw data for every dataset."""
        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in BBBC.datasets:
                threads.append(executor.submit(dataset.raw, download_path))

            for f in tqdm(
                as_completed(threads),
                desc="Downloading data",
                total=len(threads),
            ):
                f.result()
