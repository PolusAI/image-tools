from typing import List, Dict, Union, Optional
import shutil
import os
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

from polus.plugins.utils.bbbc_download.download import download, get_url, remove_macosx
from polus.plugins.utils.bbbc_download.mapping import *

import pydantic
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import bs4
from bfio import BioWriter
import vaex
from skimage import io
import pyarrow as pa
import pyarrow.parquet as pq



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
    }
)


class Metadata(pydantic.BaseModel):
    """Class that contains information about a dataset's metadata."""

    path: Path
    name: str

    @pydantic.root_validator()
    @classmethod
    def valid_data(cls, values: dict) -> dict:
        if not values["path"].exists():
            raise ValueError("No metadata")

        return values

    @property
    def size(self) -> int:
        """Returns the size of the dataset's metadata in bytes."""

        raw_path = root.joinpath(self.name, "raw/Metadata")
        standard_path = root.joinpath(self.name, "standard/Metadata")
        raw_sum = sum(os.path.getsize(file) for file in raw_path.rglob("*"))
        standard_sum = sum(os.path.getsize(file) for file in standard_path.rglob("*"))

        return raw_sum + standard_sum


class GroundTruth(pydantic.BaseModel):
    """Class that contains information about a dataset's ground truth."""

    path: Path
    name: str

    @pydantic.root_validator()
    @classmethod
    def valid_data(cls, values: dict) -> dict:
        if not values["path"].exists():
            raise ValueError("No ground truth")

        return values

    @property
    def size(self) -> int:
        """Returns the size of the dataset's ground truth in bytes."""

        raw_path = root.joinpath(self.name, "raw/Ground Truth")
        standard_path = root.joinpath(self.name, "standard/Ground Truth")
        raw_sum = sum(os.path.getsize(file) for file in raw_path.rglob("*"))
        standard_sum = sum(os.path.getsize(file) for file in standard_path.rglob("*"))

        return raw_sum + standard_sum


class Images(pydantic.BaseModel):
    """Class that contains information about a dataset's images."""

    path: Path
    name: str

    @pydantic.root_validator()
    @classmethod
    def valid_data(cls, values: dict) -> dict:
        if not values["path"].exists():
            raise ValueError("No images")

        return values

    @property
    def size(self) -> int:
        """Returns the size of the dataset's images in bytes."""

        raw_path = root.joinpath(self.name, "raw/Images")
        standard_path = root.joinpath(self.name, "standard/Images")
        raw_sum = sum(os.path.getsize(file) for file in raw_path.rglob("*"))
        standard_sum = sum(os.path.getsize(file) for file in standard_path.rglob("*"))

        return raw_sum + standard_sum


class BBBCDataset(pydantic.BaseModel):
    """Class that models a BBBC dataset.

    Attributes:
        name: The name of the dataset.
        images: An Images object that contains information about the dataset's images
        ground_truth: A GroundTruth object that contains information about the dataset's ground truth
        metadata: A Metadata object that contains information about the dataset's metadata
    """

    name: str
    images: Optional[Images] = None
    ground_truth: Optional[GroundTruth] = None
    metadata: Optional[Metadata] = None
    output_path: Optional[Path]= None

    @pydantic.validator("name")
    @classmethod
    def valid_name(cls, v: str) -> str:
        """Validates the name of the dataset.

        Args:
            v: The name of the dataset to be downloaded.

        Returns:
            The name provided if validation is successful.
        """

        if v not in list(BBBC.combined_table["Accession"]):
            raise ValueError(
                v
                + " is an invalid dataset name. Valid dataset names belong to an existing BBBC dataset."
            )

        return v

    @classmethod
    def create_dataset(cls, name: str) -> Union["BBBCDataset", None]:
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
            else:
                return BBBCDataset(name=name)
        except ValueError as e:
            print(e)

            return None

    @property
    def info(self) -> Dict[str, Union[str, np.int64]]:
        """Provides information about the dataset such as its description and total images.

        Returns:
            A dictionary that contains information about the dataset.
        """

        table = BBBC.combined_table

        row = table.loc[table["Accession"] == self.name]

        info = {
            "Description": row["Description"].values[0],
            "Mode": row["Mode"].values[0],
            "Fields per sample": row["Fields per sample"].values[0],
            "Total Fields": row["Total Fields"].values[0],
            "Total Images": row["Total Images"].values[0],
            "Ground truth types": self._ground_truth_types(),
        }

        return info

    @property
    def size(self) -> int:
        """Returns the size of the dataset in bytes."""

        dataset_path = self.output_path.joinpath("BBBC",self.name)

        return sum(os.path.getsize(file) for file in dataset_path.rglob("*"))

    def _ground_truth_types(self) -> List[str]:
        """Provides the types of ground truth used by the dataset.

        Returns:
            A list of strings where each string is a type of ground truth.
        """

        res = requests.get("https://bbbc.broadinstitute.org/image_sets")
        soup = bs4.BeautifulSoup(res.content, "html.parser")
        types = []

        for t in soup.find_all("table")[:3]:
            for row in t.find_all("tr"):
                cols = row.find_all("td")

                if len(cols) > 0 and cols[0].text == self.name:
                    for link in cols[6].find_all("a"):
                        types.append(link.attrs["href"].split("#")[-1])

                    return types

    def _init_data(self,download_path:Path) -> None:
        """Initializes the images, ground_truth, and metadata attributes of the dataset."""
        download_path=download_path.joinpath("BBBC")

        images_path = download_path.joinpath(self.name, "raw/Images")
        truth_path = download_path.joinpath(self.name, "raw/Ground Truth")
        meta_path = download_path.joinpath(self.name, "raw/Metadata")

        try:
            self.images = Images(path=images_path, name=self.name)
        except ValueError:
            pass

        try:
            self.ground_truth = GroundTruth(path=truth_path, name=self.name)
        except ValueError:
            pass

        try:
            self.metadata = Metadata(path=meta_path, name=self.name)
        except ValueError:
            pass

        if self.images == None:
            print(self.name + " has no images.")

        if self.ground_truth == None and self.metadata == None:
            print(self.name + " has no ground truth or metadata.")

        return

    def raw(self,download_path: Path) -> None:
        """Download the dataset's raw data."""
        self.output_path=download_path

        download(self.name,download_path)
        self._init_data(download_path)

        return

    def standard(self, extension: str) -> None:
        """Standardize the dataset's raw data.

        Args:
            extension: The extension of the standard image. Can be ".ome.tif" or ".ome.zarr".
        """

        if extension not in [".ome.tif", ".ome.zarr"]:
            print(
                f"ERROR: {extension} is an invalid extension for standardization. Must be .ome.tif or .ome.zarr."
            )
            return

        if self.images == None:
            print(
                f"ERROR: Images for {self.name} have not been downloaded so they cannot be standardized."
            )
            return

        standard_folder = Path(root, self.name, "standard")
        arrow_file = Path("arrow", self.name + ".arrow")
        arrow_table = pq.read_table(arrow_file)
        df = vaex.from_arrow_table(arrow_table)

        if not standard_folder.exists():
            standard_folder.mkdir(parents=True, exist_ok=True)

        for i, row in df.iterrows():
            func = globals()[self.name + "_mapping"]
            out_file = func(row, extension)
            raw_image = io.imread(row["Path"])
            num_channels = 1 if len(raw_image.shape) == 2 else raw_image.shape[2]

            if row["Image Type"] == "Intensity":
                sub_folder = "Images"
            elif row["Image Type"] == "Ground Truth":
                sub_folder = "Ground Truth"
            elif row["Image Type"] == "Metadata":
                sub_folder = "Metadata"
            else:
                print("ERROR: Invalid value for attribute Image Type")
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

        print(f"Finished standardizing {self.name}")

        return


class BBBC019(BBBCDataset):
    def raw(self,download_path:Path) -> None:
        download(self.name,download_path)
        self.output_path=download_path
        save_location=download_path.joinpath("BBBC")

        # Separate images from ground truth
        save_location = save_location.joinpath("BBBC019")
        images_folder = save_location.joinpath("raw/Images")
        truth_folder = save_location.joinpath("raw/Ground Truth")
        for set in [
            x
            for x in images_folder.iterdir()
            if x.name not in [".DS_Store", "__MACOSX"]
        ]:
            for obj in [
                x
                for x in set.iterdir()
                if x.name not in ["images", "measures.mat", "desktop.ini", ".DS_Store"]
            ]:
                src = images_folder.joinpath(set.name, obj.name)
                dst = truth_folder.joinpath(set.name, obj.name)

                if dst.exists():
                    try:
                        shutil.rmtree(src)
                    except NotADirectoryError as e:
                        print(e)
                else:
                    shutil.move(src, dst)


        self._init_data(download_path)

        return


class BBBC029(BBBCDataset):
    def raw(self,download_path:Path) -> None:
        print("Started downloading BBBC029")
        self.output_path=download_path
        save_location=download_path.joinpath("BBBC")

        save_location = save_location.joinpath("BBBC029", "raw")

        if not save_location.exists():
            save_location.mkdir(parents=True, exist_ok=True)

        file_path = save_location.joinpath("Images")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC029/images.zip",
            file_path,
            "BBBC029",
        )

        file_path = save_location.joinpath("Ground Truth")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC029/ground_truth.zip",
            file_path,
            "BBBC029",
        )

        print("BBBC029 has finished downloading")
        images_folder=save_location.joinpath("Images")
        truth_folder=save_location.joinpath("Ground Truth")
        remove_macosx("BBBC029",images_folder)
        remove_macosx("BBBC029",truth_folder)
        source_directory=images_folder.joinpath("images")
        for source_file in source_directory.glob("*"):
            destination_file = images_folder / source_file.name
            shutil.move(source_file, destination_file)
        shutil.rmtree(source_directory)   

        source_directory=truth_folder.joinpath("ground_truth")
        for source_file in source_directory.glob("*"):
            destination_file = truth_folder / source_file.name
            shutil.move(source_file, destination_file)
        shutil.rmtree(source_directory)  

        self._init_data(download_path)

        return


class BBBC041(BBBCDataset):
    def raw(self,download_path:Path) -> None:
        download(self.name,download_path)
        self.output_path=download_path
        save_location=download_path.joinpath("BBBC")

        # Separate images from ground truth
        save_location = save_location.joinpath("BBBC041")
        file_names = ["test.json", "training.json"]

        if not save_location.joinpath("raw/Ground Truth").exists():
            save_location.joinpath("raw/Ground Truth").mkdir(
                parents=True, exist_ok=True
            )

        for file in file_names:
            src = save_location.joinpath("raw/Images/malaria", file)
            dst = save_location.joinpath("raw/Ground Truth")

            if dst.joinpath(file).exists():
                os.remove(src)
            else:
                shutil.move(src, dst)

        self._init_data(download_path)

        return


class BBBC042(BBBCDataset):
    def raw(self,download_path:Path) -> None:
        print("Started downloading BBBC042")
        self.output_path=download_path
        save_location=download_path.joinpath("BBBC")

        save_location = save_location.joinpath("BBBC042", "raw")

        if not save_location.exists():
            save_location.mkdir(parents=True, exist_ok=True)

        file_path = save_location.joinpath("Images")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC042/images.zip",
            file_path,
            "BBBC042",
        )

        file_path = save_location.joinpath("Ground Truth")
        get_url(
            "https://data.broadinstitute.org/bbbc/BBBC042/positions.zip",
            file_path,
            "BBBC042",
        )

        print("BBBC042 has finished downloading")
        images_folder=save_location.joinpath("Images")
        truth_folder=save_location.joinpath("Ground Truth")
        remove_macosx("BBBC029",images_folder)
        remove_macosx("BBBC029",truth_folder)

        self._init_data(download_path)

        return


class BBBC046(BBBCDataset):
    def raw(self, download_path: Path) -> None:
        download(self.name,download_path)
        self.output_path=download_path
        save_location=download_path.joinpath("BBBC")

        # Separate images from ground truth
        try:
            save_location = save_location.joinpath(self.name)
            images_folder = save_location.joinpath("raw/Images")
            truth_folder = save_location.joinpath("raw/Ground Truth")

            # Extract these files because they do not extract automatically
            for file in ["OE-ID350-AR-1.zip", "OE-ID350-AR-2.zip", "OE-ID350-AR-4.zip", "OE-ID350-AR-8.zip"]:
                with ZipFile(images_folder.joinpath(file), "r") as zfile:
                    zfile.extractall(images_folder)

                os.remove(images_folder.joinpath(file))

            if not truth_folder.exists():
                truth_folder.mkdir(parents=True, exist_ok=True)

            # Iterate over folders in the images folder
            for folder in images_folder.iterdir():
                if not truth_folder.joinpath(folder.name).exists():
                    truth_folder.joinpath(folder.name).mkdir(
                        parents=True, exist_ok=True
                    )

                # Move ground truth data to Ground Truth folder
                for obj in folder.iterdir():
                    if obj.name.endswith((".txt", ".tif")):
                        src = obj
                        dst = truth_folder.joinpath(folder.name, obj.name)

                        if dst.exists():
                            os.remove(src)
                        else:
                            shutil.move(src, dst)

            self._init_data(download_path)
        except Exception as e:
            print(
                "BBBC046 downloaded successfully but an error occurred when organizing raw data."
            )
            print("ERROR: " + str(e))

        return


class BBBC054(BBBCDataset):
    def raw(self, download_path:Path) -> None:
        download(self.name,download_path)
        self.output_path=download_path
        save_location=download_path.joinpath("BBBC")

        # Separate images from ground truth
        save_location = save_location.joinpath(self.name)
        src = save_location.joinpath("raw/Images", "Replicate1annotation.csv")
        dst = save_location.joinpath("raw/Ground Truth", "Replicate1annotation.csv")

        if not dst.exists():
            dst.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            os.remove(src)
        else:
            shutil.move(src, dst)

        self._init_data(download_path)

        return


class IDAndSegmentation:
    """Class that models the Identification and segmentation table on https://bbbc.broadinstitute.org/image_sets.

    Attributes:
        name: The name of the table as seen on the BBBC image set webpage
        table: The Identification and segmentation table as a pandas DataFrame
    """

    name: str = "Identification and segmentation"
    table: pd.DataFrame = tables[0]

    @classmethod
    @property
    def datasets(cls) -> List[BBBCDataset]:
        """Returns a list of all datasets in the table.

        Returns:
            A list containing a Dataset object for each dataset in the table.
        """

        return [BBBCDataset.create_dataset(name) for name in cls.table["Accession"]]

    @classmethod
    def raw(cls,download_path:Path) -> None:
        """Downloads raw data for every dataset in this table"""

        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in IDAndSegmentation.datasets:
                threads.append(executor.submit(dataset.raw(download_path)))

            for f in tqdm(
                as_completed(threads), desc=f"Downloading data", total=len(threads)
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

    @classmethod
    @property
    def datasets(cls) -> List[BBBCDataset]:
        """Returns a list of all datasets in the table.

        Returns:
            A list containing a Dataset object for each dataset in the table.
        """

        return [BBBCDataset.create_dataset(name) for name in cls.table["Accession"]]

    @classmethod
    def raw(cls,download_path:Path) -> None:
        """Downloads raw data for every dataset in this table"""

        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in PhenotypeClassification.datasets:
                threads.append(executor.submit(dataset.raw(download_path)))

            for f in tqdm(
                as_completed(threads), desc=f"Downloading data", total=len(threads)
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

    @classmethod
    @property
    def datasets(cls) -> List[BBBCDataset]:
        """Returns a list of all datasets in the table.

        Returns:
            A list containing a Dataset object for each dataset in the table.
        """

        return [BBBCDataset.create_dataset(name) for name in cls.table["Accession"]]

    @classmethod
    def raw(cls,download_path:Path) -> None:
        """Downloads raw data for every dataset in this table"""

        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in ImageBasedProfiling.datasets:
                threads.append(executor.submit(dataset.raw(download_path)))

            for f in tqdm(
                as_completed(threads), desc=f"Downloading data", total=len(threads)
            ):
                f.result()


class BBBC:
    """Class that models the Broad Bioimage Benchmark Collection (BBBC).

    BBBC has tables that contain datasets. Datasets are separated into tables
    based on how they can be used. Each dataset has images and ground truth.
    Read more about BBBC here: https://bbbc.broadinstitute.org.
    """

    @classmethod
    @property
    def datasets(cls) -> List[BBBCDataset]:
        """Returns a list of all datasets in BBBC.

        Returns:
            A list containing a Dataset object for each dataset in BBBC.
        """

        table = BBBC.combined_table

        return [BBBCDataset.create_dataset(name) for name in table["Accession"]]

    @classmethod
    @property
    def combined_table(cls) -> pd.DataFrame:
        """Combines each table on https://bbbc.broadinstitute.org/image_sets into a single table.

        Returns:
            A pandas DataFrame representation of the combined table.
        """

        # Combine each table into one table
        combined_table = (
            pd.concat(tables)
            .drop(columns=["Ground truth"])
            .drop_duplicates("Accession")
        )

        return combined_table

    @classmethod
    def raw(cls,download_path:Path) -> None:
        """Downloads raw data for every dataset."""

        num_workers = max(cpu_count(), 2)
        threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for dataset in BBBC.datasets:
                threads.append(executor.submit(dataset.raw(download_path)))

            for f in tqdm(
                as_completed(threads), desc=f"Downloading data", total=len(threads)
            ):
                f.result()
