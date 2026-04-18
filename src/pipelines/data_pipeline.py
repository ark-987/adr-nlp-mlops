from logging import config
import os
import pandas as pd
from datetime import datetime

from src.data.ingest import download_kaggle_dataset
from src.data.validation.ge_validator import validate_dataframe
from src.data.split import split_data
from src.data.clean import introduce_noise
from src.cleaning_agent import CleaningAgent


def run_data_pipeline(config):

    raw_path = config["paths"]["raw_data"]
    interim_path = config["paths"]["interim_data"]
    processed_path = config["paths"]["processed_data"]

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(interim_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # -------------------------
    # 1. INGESTION
    # -------------------------
    download_kaggle_dataset(config)

    df = pd.read_csv(raw_path + config["dataset"]["train_file"])

    validate_dataframe(df, "raw")

    df.to_csv(raw_path + f"raw_{timestamp}.csv", index=False)

    # -------------------------
    # 2. NOISE SIMULATION
    # -------------------------
    text_col = config["dataset"]["text_column"]


    if config["data"]["simulate_noise"]:
       df[text_col] = df[text_col].apply(introduce_noise) 

    validate_dataframe(df, "noisy")

    df.to_csv(interim_path + f"noisy_{timestamp}.csv", index=False)

    # -------------------------
    # 3. AI AGENT CLEANING
    # -------------------------
    if config["agent"]["enabled"]:
        agent = CleaningAgent(config)
        df[text_col] = df[text_col].apply(agent.clean)

    validate_dataframe(df, "cleaned")

    df.to_csv(interim_path + f"cleaned_{timestamp}.csv", index=False)

# -------------------------
# 4. SPLIT DATA
# -------------------------
    train_df, val_df, test_df = split_data(df, config)

# -------------------------
# 5. SAVE FINAL DATASETS
# -------------------------
# Timestamped (for tracking)
    train_df.to_csv(processed_path + f"train_{timestamp}.csv", index=False)
    val_df.to_csv(processed_path + f"val_{timestamp}.csv", index=False)
    test_df.to_csv(processed_path + f"test_{timestamp}.csv", index=False)

# Fixed filenames (for training)
    train_df.to_csv(processed_path + "train.csv", index=False)
    val_df.to_csv(processed_path + "val.csv", index=False)
    test_df.to_csv(processed_path + "test.csv", index=False)

    print("Data pipeline completed successfully")
    return train_df, val_df, test_df