import kaggle
import os
import zipfile


def unzip_all(zip_name: str):
    with zipfile.ZipFile(f"data/{zip_name}.zip") as zip_file:
        zip_file.extractall(path="data/")


def download():
    print("Downloading files")
    competition = "histopathologic-cancer-detection"
    kaggle.KaggleApi().authenticate()
    kaggle.api.competition_download_cli(competition, path="data/")
    unzip_all(competition)
    os.remove(f"data/{competition}.zip")


if os.path.exists("data/"):
    csv_files = len([f for f in os.listdir("data/") if f.endswith(".csv")])
    if csv_files < 2:
        download()

else:
    os.mkdir("data/")
    download()

print("Data loaded")

