import os
import pandas as pd
from src.ingest import download_kaggle_dataset
from src.ge_expectations import create_suite
from src.gcs_utils import upload_directory_to_gcs
from src.cleaning_agent import CleaningAgent
from src.split import split_data
# Import the centralized production config loader
from src.config_loader import load_config 


def run_data_pipeline(config=None):
    print("========================================")
    print("   STARTING END-TO-END DATA PIPELINE    ")
    print("========================================\n")

    if config is None:
        config = load_config()

    # 1. Ingest Data & Raw GCS Sync
    print("--- Step 1: Data Ingestion & Raw GCS Sync ---")
    download_kaggle_dataset(config)

    # 2. Read Ingested Data safely (UPDATED: Moved above step 2 to process bad lines first)
    raw_path = config["paths"]["raw_data_file"]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Critical Error: Raw file missing at {raw_path}")
        
    print("\n--- Reading Data and Filtering Formatting Anomalies ---")
    # UPDATED: Added on_bad_lines='skip' to bypass CSV line field breaks safely
    df = pd.read_csv(raw_path, on_bad_lines='skip')
    text_col = config["dataset"].get("text_column", "review")

    # 3. Validate Data with Great Expectations
    print("\n--- Step 2: Data Quality Validation ---")
    # UPDATED: We now pass the cleaned in-memory dataframe 'df' directly into the suite
    create_suite(df)

    # 4. Cleaning Agent Processing
    print("\n--- Step 3: Cleaning Agent Processing ---")
    agent = CleaningAgent(config)
    df[text_col] = df[text_col].apply(agent.clean)

    os.makedirs(os.path.dirname(config["paths"]["cleaned_data_file"]), exist_ok=True)
    df.to_csv(config["paths"]["cleaned_data_file"], index=False)
    print(f"Cleaned dataset saved locally to: {config['paths']['cleaned_data_file']}")

    # 5. Generate Splits & Save train.csv / val.csv / test.csv
    print("\n--- Step 4: Data Splitting (Train/Val/Test) ---")
    train_df, val_df, test_df = split_data(df, config)

    # FIXED: Ensure the local target folder exists and write the CSV files to your hard drive
    split_dir = config["paths"]["split_dir"]
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Persisting data split partitions locally to: {split_dir}")
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    print("Files successfully saved to disk.")

    # 6. Upload Finished Splits Folder to GCS
    print("\n--- Step 5: Uploading Preprocessed Data to GCS ---")
    gcs_bucket = config["gcp"]["bucket_name"]
    gcs_dest_dir = config["gcp"]["processed_gcs_dir"]
    upload_directory_to_gcs(split_dir, gcs_bucket, gcs_dest_dir)

    print("\n========================================")
    print("  DATA PIPELINE EXECUTION SUCCEEDED   ")
    print("========================================")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_data_pipeline()
