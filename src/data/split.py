from sklearn.model_selection import train_test_split

def split_data(df, config):

    test_size = config["split"]["test_size"]
    val_size = config["split"]["val_size"]

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

    return train_df, val_df, test_df