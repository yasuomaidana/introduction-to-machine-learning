import shutil

import kaggle
import os
import zipfile


def unzip_by_type(name: str, file_type: str):
    with zipfile.ZipFile(f"data/{name}.zip", 'r') as zip_file:
        for file_info in zip_file.infolist():
            if file_info.filename.endswith(f".{file_type}"):
                zip_file.extract(file_info, path=f"data/")


def moves_from_inner_dir(temp_dir):
    temp_dir = "data/" + temp_dir
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(os.path.dirname(root), file)
            shutil.move(src_file, dst_file)
    os.rmdir(temp_dir)


def clean_dirs():
    for _, dirs, _ in os.walk("data/"):
        for directory in dirs:
            moves_from_inner_dir(directory)


dat_files = [f for f in os.listdir("data/") if f.endswith(".dat")]
csv_files = [f for f in os.listdir("data/") if f.endswith(".csv")]
if len(dat_files) < 3:
    print("Downloading kaggle files")
    kaggle.api.authenticate()
    dataset = "movielens-1m-dataset"
    kaggle.api.dataset_download_cli("odedgolden/" + dataset, path="data/")
    unzip_by_type(dataset, "dat")
if len(csv_files) < 4:
    print("Unzipping files")
    unzip_by_type("data", "csv")
    clean_dirs()
print("Data loaded")
