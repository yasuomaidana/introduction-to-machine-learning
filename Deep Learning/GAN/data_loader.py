import kaggle
import os
import zipfile


def get_directory_size(directory_path):
    total_size = 0
    for entry in os.scandir(directory_path):
        if entry.is_file():
            total_size += entry.stat().st_size
        elif entry.is_dir():
            total_size += get_directory_size(entry.path)
    return total_size


def unzip_all(zip_name: str):
    with zipfile.ZipFile(f"data/{zip_name}.zip") as zip_file:
        zip_file.extractall(path="data/")


def download(competition):
    print("Downloading files")
    kaggle.KaggleApi().authenticate()
    kaggle.api.competition_download_cli(competition, path="data/")
    unzip_all(competition)
    os.remove(f"data/{competition}.zip")


if os.path.exists("data/"):
    size = get_directory_size("data/")
    if size != 385866949:
        download("gan-getting-started")

else:
    os.mkdir("data/")
    download("gan-getting-started")

print("Data loaded")
