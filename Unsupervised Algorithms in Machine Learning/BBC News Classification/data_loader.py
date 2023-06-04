import kaggle
import os
import zipfile


def unzip_all(zip_name: str):
    with zipfile.ZipFile(f"data/{zip_name}.zip") as zip_file:
        zip_file.extractall(path="data/")


csv_files = len([f for f in os.listdir("data/") if f.endswith(".csv")])
if csv_files < 3:
    print("Downloading files")
    competition = "learn-ai-bbc"
    kaggle.KaggleApi().authenticate()
    kaggle.api.competition_download_cli(competition, path="data/")
    unzip_all(competition)
    os.remove(f"data/{competition}.zip")
print("Data loaded")
