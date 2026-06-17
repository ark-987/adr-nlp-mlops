import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import MLflow layers globally
import mlflow
import mlflow.pytorch

# Custom modular pipeline imports
from src.build_features import load_ner_pipeline, enrich_and_label_batched
from src.config_loader import load_config

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        
        # FIX: Explicitly cast labels to .long() to prevent floating point errors
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1).long())
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def upload_directory_to_gcs(local_path, bucket_name, gcs_path):
    """Placeholder helper matching your custom direct GCP bucket backup function"""
    print(f"[GCP] Streaming data directories up to gs://{bucket_name}/{gcs_path}...")
    pass

def run_training_pipeline(config, train_df=None, val_df=None, test_df=None):
    text_col = config["dataset"].get("text_column", "review")
    target_col = config["dataset"]["target_column"]
    output_dir = config["paths"]["models"]
    bucket_name = config["gcp"]["bucket_name"]
    model_name = config["training_model_name"] 

    # Point MLflow to tracking server dynamically from config
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Ensure tracking parent directory exists locally
    os.makedirs(output_dir, exist_ok=True)
    
    # Conditional loading logic based on environment profile settings
    if train_df is None:
        run_mode = config.get("mode", "test_dummy")
        if run_mode == "test_dummy":
            print("[INFO] Running low-memory local dummy fallback mode...")
            dummy_path = config["paths"]["dummy_data"]
            # skips broken rows and auto-adjusts for mismatched quotes
            raw_df = pd.read_csv(dummy_path, on_bad_lines='skip', sep=None, engine='python')

            # Create a rapid tiny split layout from the 20-review seed
            train_df = raw_df.sample(frac=0.6, random_state=42)
            val_df = raw_df.drop(train_df.index).sample(frac=0.5, random_state=42)
            test_df = raw_df.drop(train_df.index).drop(val_df.index)
        else:
            print("[INFO] Loading complete processed production datasets from disk splits...")
            split_path = config["paths"]["split_dir"]
            train_df = pd.read_csv(os.path.join(split_path, "train.csv"))
            val_df = pd.read_csv(os.path.join(split_path, "val.csv"))
            test_df = pd.read_csv(os.path.join(split_path, "test.csv"))

    print(f"[INFO] Initializing sequence classification architecture backend: {model_name}")

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
    
    # 2. Map dataset features using low batch sizes to stay under local memory ceilings
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
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2]),
        }

    # No checkpoint saving completely to protect 5.3GB drive space
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training_epochs"],      
        per_device_train_batch_size=config["training_batch_size"], 
        per_device_eval_batch_size=config["training_batch_size"],  
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

        # Extraction parameters for Model Registration 
        registered_name = config["mlflow"].get("registered_model_name", "biobert_adr_classifier")
        run_id = run.info.run_id

        # MLOPS ZERO-DISK FIX: Streams weights directly out of system RAM over network to cloud registry
        print("[MLOPS FIXED] Streaming model weights directly from RAM to MLflow Cloud Server...")
        mlflow.pytorch.log_model(
            pytorch_model=trainer.model,
            artifact_path="model",
            registered_model_name=registered_name
        )
        print(f"Model successfully streamed and registered in cloud registry as: {registered_name}")
        
        # Direct GCP backup pipeline channel trigger
        #print("Uploading model artifacts to Google Cloud Storage...")
        #upload_directory_to_gcs(output_dir, bucket_name, config["gcp"]["model_gcs_file"])

        # --- AUTOMATED PRODUCTION OVERWRITE PROTECTION when uploading artifacts to Google Cloud Storage ---
        model_gcs_file = config["gcp"]["model_blob_path"]  

        try:
            import subprocess
            # Get the name of the current active git branch
            branch_cmd = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                        capture_output=True, text=True, check=True)
            active_branch = branch_cmd.stdout.strip()
        except Exception:
            # Fallback to test mode if git command fails for any reason
            active_branch = "local-development"

        # Safe Guard: Force test naming unless explicitly running on 'main' or via production flags
        if active_branch != "main" and "--prod" not in sys.argv:
            original_file = model_gcs_file
            # If model_gcs_file is "final_model.zip", this changes it to "final_model-test.zip"
            if "." in model_gcs_file:
                name_parts = model_gcs_file.rsplit(".", 1)
                model_gcs_file = f"{name_parts[0]}-test.{name_parts[1]}"
            else:
                model_gcs_file = f"{model_gcs_file}-test"
                
            print(f"[SAFETY ACTIVE] Non-production branch '{active_branch}' detected!")
            print(f"[SAFETY ACTIVE] Diverting upload path from '{original_file}' to '{model_gcs_file}' to protect production.")

        # Trigger the upload safely
        print(f"Uploading model artifacts to Google Cloud Storage as: {model_gcs_file}...")
        upload_directory_to_gcs(output_dir, bucket_name, model_gcs_file)


if __name__ == "__main__":
    current_config = load_config()
    run_training_pipeline(current_config)
