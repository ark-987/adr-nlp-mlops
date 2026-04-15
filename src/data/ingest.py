from kaggle.api.kaggle_api_extended import KaggleApi
import os

def download_kaggle_dataset(config):
    dataset = config["dataset"]["name"]
    path = config["paths"]["raw_data"]

    os.makedirs(path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=path, unzip=True)

