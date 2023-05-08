from zipfile import ZipFile
import os
import shutil

if not os.path.exists("data/data.csv"):
    with(ZipFile("data/data.zip")) as data_zip:
        print("Unzipping data")
        data_zip.extractall("data/")
    for file in os.listdir("data/Files"):
        shutil.move("data/Files/" + file, "data/" + file)
    os.rmdir("data/Files")
print("Data loaded")
