import os
import shutil
from dotenv import load_dotenv
from src.gcs_utils import upload_to_gcs

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


def download_kaggle_dataset(config):
    raw_dir = config["paths"]["raw_data_dir"]
    target_file = config["paths"]["raw_data_file"]
    dummy_file = config["paths"]["dummy_data"]
    
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Handle Ingestion Data Source
    if os.path.exists(dummy_file):
        print(f"Bypassing Kaggle download. Using local dummy file: {dummy_file}")
        shutil.copy(dummy_file, target_file)
        print(f"Successfully staged dummy data to {target_file}")
    else:
        if not KAGGLE_AVAILABLE:
            raise ImportError(
                "Critical Error: Dummy file missing and 'kaggle' package is unavailable. "
                "Cannot proceed with pipeline ingestion."
            )
            
        print("Dummy file not found. Falling back to Kaggle API download...")
        load_dotenv()
        os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
        os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

        dataset = config["dataset"]["name"]
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=raw_dir, unzip=True)
        
        downloaded_expected = os.path.join(raw_dir, "drugsComTrain_raw.csv") 
        if os.path.exists(downloaded_expected) and downloaded_expected != target_file:
            shutil.move(downloaded_expected, target_file)

    # 2. Sync Raw Data to Google Cloud Storage
    bucket = config["gcp"]["bucket_name"]
    destination_blob = config["gcp"]["raw_gcs_path"]
    upload_to_gcs(target_file, bucket, destination_blob)
