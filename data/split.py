from sklearn.model_selection import train_test_split

def split_data(df, config):
    
    # Ensure keys match pipeline.yaml exactly
    test_size = config["split"]["test_size"]
    val_size = config["split"]["val_size"]
    output_dir = config["paths"]["processed_data"]


    target_col = config["dataset"].get("target_column", None)

    stratify_col = df[target_col] if target_col and target_col in df.columns else None

    # First split
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=42,
        stratify=stratify_col
    )

    # Adjust stratify for second split
    stratify_temp = temp_df[target_col] if target_col else None

    val_ratio = val_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio,
        random_state=42,
        stratify=stratify_temp
    )
    # --- New Saving Logic ---
    # Define the target directory (usually data/processed or similar)
    output_dir = config["paths"].get("processed_data", "data/processed")
    os.makedirs(output_dir, exist_ok=True) # Creates folder if it doesn't exist

    # Map names to their dataframes for a clean loop
    datasets = {
        "train.csv": train_df,
        "val.csv": val_df,
        "test.csv": test_df
    }

    for filename, data in datasets.items():
        save_path = os.path.join(output_dir, filename)
        data.to_csv(save_path, index=False) # index=False prevents extra columns
        print(f"Saved: {save_path}")

    return train_df, val_df, test_df