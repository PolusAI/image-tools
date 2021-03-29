from pathlib import Path
import requests, zipfile

""" Get an example image """
# Set up the directories
PATH = Path(__file__).with_name('data')
PATH.mkdir(parents=True, exist_ok=True)

# Download the data if it doesn't exist
URL = "https://github.com/USNISTGOV/MIST/wiki/testdata/"
FILENAME = "Small_Fluorescent_Test_Dataset.zip"
if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
    
with zipfile.ZipFile(PATH/FILENAME, 'r') as zip_ref:
    zip_ref.extractall(PATH)