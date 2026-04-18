from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import os

def download_kaggle_dataset(config):
    load_dotenv()

    
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
    

    dataset = config["dataset"]["name"]
    path = config["paths"]["raw_data"]

    os.makedirs(path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=path, unzip=True)