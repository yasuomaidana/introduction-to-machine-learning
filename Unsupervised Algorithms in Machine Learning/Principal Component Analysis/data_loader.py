from zipfile import ZipFile
import glob

if not glob.glob(r"data/train_images*.pkl"):
    print("Extracting")
    with ZipFile("data/data.zip") as data_zip:
        data_zip.extractall("data/")
