import os


if not os.path.exists("./data/house-prices"):
    print("Unzipping data")
    import zipfile
    with zipfile.ZipFile("./data/house-prices-advanced-regression-techniques.zip") as zip_ref:
        zip_ref.extractall("./data/house-prices")
        zip_ref.close()
    print("Data unzipped")
