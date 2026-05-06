import os
import yaml
import torch
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# MLflow & Optuna integrations
import mlflow
import optuna

# Import your feature builder function
from src.features.build_features import load_ner_pipeline, enrich_and_label_batched


# 1. CUSTOM TRAINER FOR CLASS WEIGHTS
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


# 2. EVALUATION METRICS FUNCTION
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
    bucket_name = "adr-nlp"

    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # 3. LOAD DATA FROM GCS
    if train_df is None:
        print(" Pulling processed training files directly from GCS...")
        try:
            train_df = pd.read_csv(f"gs://{bucket_name}/data/processed/train.csv")
            val_df = pd.read_csv(f"gs://{bucket_name}/data/processed/val.csv")
            test_df = pd.read_csv(f"gs://{bucket_name}/data/processed/test.csv")
        except Exception as e:
            raise e

    # 4. CALCULATE CLASS WEIGHTS
    labels = train_df[target_col].values
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    # 5. DATASET PREPARATION & ENRICHMENT
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    ner_pipeline = load_ner_pipeline()
    train_ds = train_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=32)
    val_ds = val_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=32)
    test_ds = test_ds.map(lambda x: enrich_and_label_batched(x, ner_pipeline), batched=True, batch_size=32)

    # 6. TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(examples):
        return tokenizer(examples[text_col], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize_func, batched=True)
    val_ds = val_ds.map(tokenize_func, batched=True)
    test_ds = test_ds.map(tokenize_func, batched=True)

    train_ds = train_ds.rename_column(target_col, "labels")
    val_ds = val_ds.rename_column(target_col, "labels")
    test_ds = test_ds.rename_column(target_col, "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 7. OPTUNA MODEL INITIALISER
    # Required for hyperparameter tuning to spawn fresh instances [2, 3]
    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define hyperparameter search space [1]
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        }

    # 8. TRAINING WITH OPTUNA & MLFLOW
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_epoch=True,
        metric_for_best_model="f1"
    )

    trainer = CustomTrainer(
        class_weights=weights_tensor,
        model_init=model_init, # Notice we pass the function instead of a loaded model [2, 3]
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    # Wrap hyperparameter search in MLflow context
    with mlflow.start_run() as run:
        print("Starting Optuna hyperparameter search...")
        
        # Search for the best parameters across 5 different runs [1]
        # Direction  'maximize' because we look at F1 score
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=5
        )
        
        print("Best hyperparameter run found:", best_run.hyperparameters)
        
        # Log the best parameters directly to MLflow
        mlflow.log_params(best_run.hyperparameters)

        # Apply the best parameters to our trainer and train the final model [3]
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
            
        print("Training final model with best found parameters...")
        trainer.train()

        print("Evaluating on final test set...")
        test_results = trainer.predict(test_ds)
        
        # Log Final Test Metrics to MLflow
        mlflow.log_metrics(test_results.metrics)
        print("Final Test Metrics:", test_results.metrics)

                # 8.5 SAVE METRICS FOR DVC
        import json
        os.makedirs("logs", exist_ok=True)
        with open("logs/metrics.json", "w") as f:
            json.dump(test_results.metrics, f, indent=4)


        # 9. STORE LOCALLY AND PUSH TO GCS
        print("Saving model locally...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir) # Ensuring tokenizer maps alongside saved model files
              
        # --- NEW: REGISTER MODEL TO MLFLOW REGISTRY ---
        # This makes the model_uri "models:/biobert_adr_classifier/latest" work!
        registered_name = config["mlflow"].get("registered_model_name", "biobert_adr_classifier")
        run_id = run.info.run_id
        
        print(f"Registering model as '{registered_name}' in MLflow...")
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/outputs", # 'outputs' or wherever artifacts are logged
            name=registered_name)

    # Push full local MLflow runs directory and model folder to GCS Bucket
    print(f"Uploading assets to GCS bucket: {bucket_name}...")
    try:
        os.system(f"gcloud storage cp -r ./mlruns gs://{bucket_name}/mlflow_logs/")
        os.system(f"gcloud storage cp -r {output_dir} gs://{bucket_name}/models/")
        print("Model and MLflow logs successfully backed up to GCS!")
    except Exception as e:
        print(f"Failed to upload to GCS: {e}")

    print("Training complete.")
    
    return trainer.model, tokenizer

    

