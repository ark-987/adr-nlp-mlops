import os
import pandas as pd
from datetime import datetime
from data.split import split_data
from src.data.validation.ge_validator import validate_dataframe
from src.cleaning_agent import CleaningAgent  # <--- NEW IMPORT

def run_data_pipeline(config):
    raw_path = config["paths"]["raw_data"]
    processed_path = config["paths"]["processed_data"]
    text_col = config["dataset"]["text_column"]

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    bucket_name = config["gcs"]["bucket_name"]
    raw_file = config["dataset"]["train_file"]

    # 1. INGEST FROM GCS
    print(f"Ingesting: gs://{bucket_name}/raw/{raw_file}")
    local_raw_path = os.path.join(raw_path, raw_file)
    os.system(f"gcloud storage cp gs://{bucket_name}/raw/{raw_file} {local_raw_path}")
         
    df = pd.read_csv(local_raw_path)
    validate_dataframe(df, "raw")

    # -------------------------
    # 2. CLEAN DATA (AGENT)
    # -------------------------
    if config["agent"]["enabled"]:
        print(f"Applying CleaningAgent to column: {text_col}...")
        agent = CleaningAgent(config)
        df[text_col] = df[text_col].apply(agent.clean)

    # 3. SPLIT DATA
    print("Splitting data into Train/Val/Test...")
    train_df, val_df, test_df = split_data(df, config)

    # 4. SAVE & UPLOAD
    splits = {"train.csv": train_df, "val.csv": val_df, "test.csv": test_df}

    for filename, dataframe in splits.items():
        local_filepath = os.path.join(processed_path, filename)
        dataframe.to_csv(local_filepath, index=False)
                 
        gcs_dest = f"gs://{bucket_name}/{config['gcs']['processed_folder']}/{filename}"
        os.system(f"gcloud storage cp {local_filepath} {gcs_dest}")

    print("Data pipeline completed!")
    return train_df, val_df, test_df

