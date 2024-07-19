import kaggle


def load_data() -> None:
    kaggle.api.authenticate()  # Authenticates using your kaggle.json
    data_set = 'ipythonx/carvana-image-masking-png'
    kaggle.api.dataset_download_files(data_set, path='../data/raw', unzip=True)


if __name__ == "__main__":
    load_data()
    print("Data loaded successfully")
