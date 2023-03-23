import requests
import os.path
from os import path

if not path.exists("data/auto-mpg.data"):
    print("Downloading data")
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    response = requests.get(URL)
    open("data/auto-mpg.data", "wb").write(response.content)
print("Data loaded")
