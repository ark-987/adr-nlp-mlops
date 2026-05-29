import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, config: dict):
    """Splits the dataframe into Train/Val/Test data partitions safely."""
    print("Partitioning processed matrix into experimental splits...")
    
    # Extract configuration safely from the 'split' block in pipeline.yaml
    split_cfg = config.get("split", {})
        
    # Read the values present in your pipeline.yaml, fallback to defaults if missing
    val_size = split_cfg.get("val_size", 0.1)
    test_size = split_cfg.get("test_size", 0.1)
    train_size = split_cfg.get("train_size", 1.0 - (val_size + test_size))  # Automatically becomes 0.8
    
    # Step 1: Detect class distributions for dummy data compatibility
    stratify_target = None
    if "rating" in df.columns:
        min_class_count = df["rating"].value_counts().min()
        if min_class_count >= 2 and len(df) > 50:
            print("Applying stratified grouping based on target distribution scale.")
            stratify_target = df["rating"]
        else:
            print("[INFO] Small dummy matrix detected. Bypassing stratification to prevent split errors.")

    # Step 2: Calculate proportions relative to remaining data
    relative_test_size = test_size / (val_size + test_size)

    # Step 3: Run the initial extraction pass (Train vs Temporary)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=42,
        stratify=stratify_target
    )

    # Step 4: Re-evaluate stratification for the second split pass
    temp_stratify = temp_df["rating"] if stratify_target is not None else None

    # Step 5: Run the second extraction pass (Val vs Test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=42,
        stratify=temp_stratify
    )

    print(f"Data successfully split! Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")
    return train_df, val_df, test_df
