import os
import pandas as pd



def run_training_pipeline(config, train_df=None, val_df=None, test_df=None):
    import transformers
    text_col = config["dataset"].get("text_column", "review")
    target_col = config["dataset"]["target_column"]

    model_name = config["training"]["model_name"]
    output_dir = config["paths"]["models"]

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # LOAD DATA (if not passed)
    # -------------------------
    if train_df is None:
        print("📂 Loading processed data from disk...")

        processed_path = config["paths"]["processed_data"]

        train_df = pd.read_csv(processed_path + "train.csv")
        val_df = pd.read_csv(processed_path + "val.csv")
        test_df = pd.read_csv(processed_path + "test.csv")

    # -------------------------
    # TOKENIZER + MODEL
    # -------------------------
    print("Loading tokenizer and model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1  # regression
    )

    # -------------------------
    # TOKENIZATION
    # -------------------------
    def tokenize(texts):
        return tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    train_encodings = tokenize(train_df[text_col])
    val_encodings = tokenize(val_df[text_col])

    train_labels = train_df[target_col].values
    val_labels = val_df[target_col].values

    # -------------------------
    # TRAINING SETUP
    # -------------------------
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
    )

    # -------------------------
    # TRAINER
    # -------------------------
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=list(zip(train_encodings["input_ids"], train_labels)),
        eval_dataset=list(zip(val_encodings["input_ids"], val_labels)),
    )

    print("Training model...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)

    print("Training complete")