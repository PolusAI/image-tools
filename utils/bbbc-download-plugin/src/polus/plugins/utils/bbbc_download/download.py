from pathlib import Path
import re
from urllib.request import urlretrieve
from urllib.error import URLError
from zipfile import ZipFile
import logging

import bs4
import shutil
import requests

match_str = (
    "Images|Ground truth|Ground Truth|Metadata|Hand-annotated Ground Truth Images"
)
endings = (".txt", ".csv", ".tif", ".xlsx", ".xls", ".lst")
logger = logging.getLogger(__name__)

def get_lower_tags(tag: bs4.element.Tag) -> list:
    """Get all tags between the tag argument and the next tag of the same type.
    Args:
        tag: Get tags between this tag and the next tag of the same type
    """

    tags = []

    for sib in tag.find_next_siblings():
        if sib.name == tag.name:
            break
        else:
            tags.append(sib)

    return tags

def extract_nested_zips(name: str,zip_path:Path, extract_path:Path):
    """Unzip nested zip files.
    Args:
        name: Name of the dataset
        zip_path: Path to the zip file
        extract_path: The path where the unzipped files will be saved
    """

    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    zip_path.unlink()
    extracted_folder_name = zip_path.stem  # Name with .zip extension
    extracted_folder_name = extract_path.joinpath(extracted_folder_name.replace('.zip',''))
    remove_macosx(name,extract_path)
    
    nested_zip_files = list(extracted_folder_name.glob("*.zip"))

    for nested_zip_file in nested_zip_files:
        nested_extract_path = nested_zip_file.parent
        extract_nested_zips(nested_zip_file, nested_extract_path)
        

def get_url(url: str, save_location: Path, name: str) -> None:
    """Get the given url and save it.
    Args:
        url: The url to get
        save_location: The path where the files will be saved
        name: The name of the dataset that the url is associated with
    """

    file_name = url.split("/")[-1]

    for download_attempts in range(10):
        if url.endswith(endings):
            try:
                if not save_location.exists():
                    save_location.mkdir(parents=True, exist_ok=True)

                urlretrieve(url, save_location.joinpath(file_name))
            except URLError as e:
                if download_attempts == 9:
                    logger.info(f"FAILED TO DOWNLOAD {url} for {name}")
                    logger.info(f"ERROR {str(e)}")

                continue
        elif url.endswith(".zip"):
            try:
                zip_path, _ = urlretrieve(url)

                with ZipFile(zip_path, "r") as zfile:
                    zfile.extractall(save_location)
                    
                            
                                
            except URLError as e:
                if download_attempts == 9:
                    logger.info(f"FAILED TO DOWNLOAD {url} for {name}")
                    logger.info(f"ERROR {str(e)}")

                continue
            except Exception as e:
                logger.info(f"{e}")

                continue

        break

    return

def remove_macosx(name:str, location:Path)-> None:
    """ Remove the __MACOSX folder from the downlpoaded dataset.
    Args:
        name: The name of the dataset
        location: The partent directory of the __MACOSX folder.
    """
    folders=[folders for folders in location.iterdir() if folders.is_dir()]
    for f in folders:
        if f.name=="__MACOSX":
            shutil.rmtree(f)
            logger.info(f"Deleted the __MACOSX  folder in {name}")

def download(name: str,download_path:Path) -> None:
    """Download a single dataset.
    Args:
        name: The name of the dataset to be downloaded
        downlaod_path: Path to donwload the dataset
    """

    logger.info(f"Started downloading {name}")
    download_path=download_path.joinpath("BBBC")

    save_location = download_path.joinpath(name, "raw")

    if not save_location.exists():
        save_location.mkdir(parents=True, exist_ok=True)

    dataset_url = "https://bbbc.broadinstitute.org/" + name

    dataset_page = requests.get(dataset_url)
    soup = bs4.BeautifulSoup(dataset_page.content, "html.parser")

    for heading in soup.find_all("h3"):
        # Ignore headings that we aren't interested in
        if re.match(match_str, heading.text.strip()) == None:
            continue

        if heading.text.strip() == "Images":
            sub_folder = "Images"
        elif heading.text.strip() == "Metadata":
            sub_folder = "Metadata"
        else:
            sub_folder = "Ground_Truth"

        # Iterate over every tag under the current heading and above the next heading
        for tag in get_lower_tags(heading):
            links = tag.find_all("a")
            data_links = [
                l for l in links if l.attrs["href"].endswith((".zip", *endings))
            ]

            for link in data_links:
                data_url = link.attrs["href"]
                file_path = save_location.joinpath(sub_folder)

                get_url(data_url, file_path, name)

        # Manually download BBBC018 ground truth because its webpage structure is incorrect
        if name == "BBBC018" and re.match("Ground truth", heading.text.strip()):
            url = "https://data.broadinstitute.org/bbbc/BBBC018/BBBC018_v1_outlines.zip"

            file_path = save_location.joinpath(sub_folder)

            get_url(url, file_path, "BBBC018")

    logger.info(f"{name} has finished downloading")
    
    images_path=save_location.joinpath("Images")
    remove_macosx(name,images_path)
    ground_path=save_location.joinpath("Ground_Truth")
    if ground_path.exists():
        remove_macosx(name,ground_path)
    
    # unzip nested zip files
    zip_files = list(images_path.glob("**/*.zip"))
    for zip_file in zip_files:
        extract_path = zip_file.parent
        extract_nested_zips(name,zip_file, extract_path)

    return
