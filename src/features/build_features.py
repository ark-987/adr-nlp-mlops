import os
import pandas as pd
import numpy as np
import transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import your feature builder
from src.features.build_features import load_ner_pipeline, enrich_and_label_batched

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def run_training_pipeline(config, train_df=None, val_df=None, test_df=None):
    text_col = config["dataset"].get("text_column", "review")
    target_col = config["dataset"]["target_column"]
    model_name = config["training"]["model_name"]
    output_dir = config["paths"]["models"]

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # 1. LOAD DATA & BUILD FEATURES
    # -------------------------
    if train_df is None:
        print("Loading processed data from disk...")
        processed_path = config["paths"]["processed_data"]
        train_df = pd.read_csv(os.path.join(processed_path, "train.csv"))
        val_df = pd.read_csv(os.path.join(processed_path, "val.csv"))
        test_df = pd.read_csv(os.path.join(processed_path, "test.csv"))

    # Convert to Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    # Enrich text and create 'adr_label' using  build_features function
    print("Enriching datasets with Biomedical NER...")
    ner_pipeline = load_ner_pipeline()
    
    train_ds = train_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=32)
    val_ds = val_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=32)
    test_ds = test_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=32)

    # -------------------------
    # 2. TOKENIZER + MODEL (Binary Classification)
    # -------------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_func(examples):
        return tokenizer(examples[text_col], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize_func, batched=True)
    val_ds = val_ds.map(tokenize_func, batched=True)
    test_ds = test_ds.map(tokenize_func, batched=True)

    # Map target column to 'labels' for Trainer recognition
    train_ds = train_ds.rename_column(target_col, "labels")
    val_ds = val_ds.rename_column(target_col, "labels")
    test_ds = test_ds.rename_column(target_col, "labels")

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns_to_keep)
    val_ds.set_format(type="torch", columns=columns_to_keep)
    test_ds.set_format(type="torch", columns=columns_to_keep)

    # -------------------------
    # 3. TRAINING SETUP
    # -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=2e-5, # Safe BERT learning rate
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_epoch=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("Training model...")
    trainer.train()

    print("Evaluating on final test set...")
    test_results = trainer.predict(test_ds)
    print("Final Test Metrics:", test_results.metrics)

    print("Saving model...")
    trainer.save_model(output_dir)
    print("Training complete.")
