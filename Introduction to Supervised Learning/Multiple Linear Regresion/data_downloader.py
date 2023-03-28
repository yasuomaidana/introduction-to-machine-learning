import requests
import os.path
import zipfile
from os import path

if not path.exists("data"):
    os.mkdir("data")

if not path.exists("data/auto-mpg.data"):
    print("Downloading data")
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    response = requests.get(URL)
    open("data/auto-mpg.data", "wb").write(response.content)
if not path.exists("data/bodyfat.csv"):
    print("Unziping files")
    with zipfile.ZipFile("extra_data.zip",'r') as zip_ref:
        zip_ref.extractall("data/")
    allfiles = os.listdir("./data/Files")
    for i in allfiles:
        os.rename("./data/Files/"+i,"./data/"+i)
    os.rmdir("./data/Files")
print("Data loaded")
