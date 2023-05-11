import kaggle
import os
import zipfile

files = [f for f in os.listdir("data/") if f.endswith(".dat")]
if len(files) < 3:
    print("Downloading files")
    kaggle.api.authenticate()
    dataset = "movielens-1m-dataset"
    kaggle.api.dataset_download_cli("odedgolden/"+dataset, path="data/")
    with zipfile.ZipFile("data/"+dataset+".zip", 'r') as zipfile:
        for file_info in zipfile.infolist():
            if file_info.filename.endswith(".dat"):
                zipfile.extract(file_info, "data/")
print("Data loaded")

