import os
import zipfile

if not os.path.exists("./data/house-prices"):
    print("Unzipping house-prices data")
    with zipfile.ZipFile("./data/house-prices-advanced-regression-techniques.zip") as zip_ref:
        zip_ref.extractall("./data/house-prices")
        zip_ref.close()
    print("Data unzipped")
if not os.path.exists("./data/house-sales"):
    print("Unzipping house-sales data")
    with zipfile.ZipFile("./data/house-sales.zip") as zip_ref:
        zip_ref.extractall("./data/house-sales")
        zip_ref.close()
    print("Data unzipped")
