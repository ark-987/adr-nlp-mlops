import os
# FORCE Hugging Face and Optuna backends to use traditional PyTorch serialization format globally
os.environ["HF_HUB_DISABLE_SAFETENSORS"] = "1"

import json
import shutil
import torch
import numpy as np
import pandas as pd

import transformers
from datasets import Dataset
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import mlflow
from src.build_features import load_ner_pipeline, enrich_and_label_batched
from src.gcs_utils import upload_directory_to_gcs # Reusing your GCS utility file


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def run_training_pipeline(config, train_df=None, val_df=None, test_df=None):
    text_col = config["dataset"].get("text_column", "review")
    target_col = config["dataset"]["target_column"]
    output_dir = config["paths"]["models"]
    bucket_name = config["gcp"]["bucket_name"]
    model_name = config["training"]["model_name"] 

    # Point MLflow to your remote tracking server
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    os.makedirs(output_dir, exist_ok=True)
    
    if train_df is None:
        print("Loading processed data from disk...")
        split_path = config["paths"]["split_dir"]
        train_df = pd.read_csv(os.path.join(split_path, "train.csv"))
        val_df = pd.read_csv(os.path.join(split_path, "val.csv"))
        test_df = pd.read_csv(os.path.join(split_path, "test.csv"))

    model_name = config["training"]["model_name"]
    print(f"[INFO] Initializing production architecture: {model_name}")

    unique_labels = sorted(train_df[target_col].unique())
    label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    
    for dataframe in [train_df, val_df, test_df]:
        dataframe[target_col] = dataframe[target_col].map(label_to_id)

    labels_array = train_df[target_col].values
    classes_present = np.unique(labels_array)
    computed_weights = compute_class_weight(class_weight="balanced", classes=classes_present, y=labels_array)
    
    full_weights_array = np.ones(num_labels, dtype=np.float32)
    for idx, cls_id in enumerate(classes_present):
        full_weights_array[cls_id] = computed_weights[idx]
    weights_tensor = torch.tensor(full_weights_array, dtype=torch.float)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    # 1. Initialize the modular adaptive NER extractor
    ner_pipeline = load_ner_pipeline()
    
    # 2. Map dataset features using your original notebook text-tagging logic
    # Using low batch sizes for local processing to stay under the 900MB memory ceiling
    print("Processing Training Data through NER mapping...")
    train_ds = train_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=2)
    print("Processing Validation Data through NER mapping...")
    val_ds = val_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=2)
    print("Processing Test Data through NER mapping...")
    test_ds = test_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=2)

    
    # Ensure this line uses the stable slow tokenizer backend
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize_func(examples):
        return tokenizer(examples[text_col], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tokenize_func, batched=True)
    val_ds = val_ds.map(tokenize_func, batched=True)
    test_ds = test_ds.map(tokenize_func, batched=True)

    train_ds = train_ds.rename_column(target_col, "labels")
    val_ds = val_ds.rename_column(target_col, "labels")
    test_ds = test_ds.rename_column(target_col, "labels")

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns_to_keep)
    val_ds.set_format(type="torch", columns=columns_to_keep)
    test_ds.set_format(type="torch", columns=columns_to_keep)

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 1),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2]), # <-- FIXED: Added [2]
        }

    # FIXED: Deactivated checkpoint tracking to bypass background trial crashes
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=2e-5, 
        eval_strategy="no",          
        save_strategy="no",          
        logging_dir="./logs",
        load_best_model_at_end=False, 
        metric_for_best_model="f1"
    )

    trainer = CustomTrainer(
        class_weights=weights_tensor,
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    with mlflow.start_run() as run:
        print("Running remote Optuna trials tracked in cloud...")
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=1 
        )
        
        mlflow.log_params(best_run.hyperparameters)

        for param_name, param_value in best_run.hyperparameters.items():
            setattr(trainer.args, param_name, param_value)
            
        trainer.train()
        test_results = trainer.predict(test_ds)
        mlflow.log_metrics(test_results.metrics)

        # Local log metric output required for DVC tracking layer checks
        os.makedirs("logs", exist_ok=True)
        with open("logs/metrics.json", "w") as f:
            json.dump(test_results.metrics, f, indent=4)

        # FIXED: Switch off safe_serialization to prevent Windows process MemoryErrors
        print("Saving final trained model binaries locally via direct disk mapping...")
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
      
        registered_name = config["mlflow"].get("registered_model_name", "biobert_adr_classifier")
        run_id = run.info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", registered_name)
        print(f"Model registered in MLflow with name: {registered_name}")
        print("Uploading model artifacts to Google Cloud Storage...")
        upload_directory_to_gcs(output_dir, bucket_name, f"models/{registered_name}")

# Entry point for running the training pipeline
if __name__ == "__main__":
    from src.config_loader import load_config
    current_config = load_config()
    run_training_pipeline(current_config)

