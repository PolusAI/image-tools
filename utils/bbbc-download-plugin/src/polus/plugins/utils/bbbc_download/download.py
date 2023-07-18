from pathlib import Path
import re
from urllib.request import urlretrieve
from urllib.error import URLError
from zipfile import ZipFile

import bs4
import shutil
import requests

match_str = (
    "Images|Ground truth|Ground Truth|Metadata|Hand-annotated Ground Truth Images"
)
endings = (".txt", ".csv", ".tif", ".xlsx", ".xls", ".lst")


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
                    print("FAILED TO DOWNLOAD: " + url + " for " + name)
                    print("ERROR: " + str(e))

                continue
        elif url.endswith(".zip"):
            try:
                zip_path, _ = urlretrieve(url)

                with ZipFile(zip_path, "r") as zfile:
                    zfile.extractall(save_location)
            except URLError as e:
                if download_attempts == 9:
                    print("FAILED TO DOWNLOAD: " + url + " for " + name)
                    print("ERROR: " + str(e))

                continue
            except Exception as e:
                print(e)

                continue

        break

    return

def remove_macosx(name:str, save_location:Path)-> None:
    images_path=save_location.joinpath("Images")
    folders=[folders for folders in images_path.iterdir() if folders.is_dir()]
    for f in folders:
        if f.name=="__MACOSX":
            shutil.rmtree(f)
            print("Deleted the __MACOSX folder in " + name)




def download(name: str,download_path:Path) -> None:
    """Download a single dataset.
    Args:
        name: The name of the dataset to be downloaded
    """

    print("Started downloading " + name)
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
            sub_folder = "Ground Truth"

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

    print(name + " has finished downloading")
    remove_macosx(name,save_location)

    return
