import os
import yaml
import pandas as pd
import mlflow  # Integrated for central experiment tracking tracking
from data_pipeline import run_data_pipeline
from training_pipeline import run_training_pipeline
from explainerAI import XAIEngine
from src.config_loader import load_config 


class Engine:
    def __init__(self, config):
        self.config = config

    def run_explanation_stage(self, model, tokenizer, test_df):
        print("\n--- Starting XAI Explanation Stage ---")
        
        text_col = self.config.get("dataset", {}).get("text_column", "review")
        
        if test_df is None:
            split_dir = self.config.get("paths", {}).get("split_dir", "data/processed/split")
            test_path = os.path.join(split_dir, "test.csv")
            print(f"Loading test set split for XAI from: {test_path}")
            test_df = pd.read_csv(test_path)

        xai = XAIEngine(model, tokenizer)

        sample_text = test_df[text_col].iloc[0]
        print(f"Explaining sample from column '{text_col}': {sample_text[:60]}...")

        # target output directory exists before generating reports
        os.makedirs("reports", exist_ok=True)
        report_path = "reports/shap_output.html"
        xai.explain(sample_text, save_path=report_path)
        print("XAI Stage Complete: SHAP values generated and saved.")

        # -------------------------------------------------------------
        # MLFLOW ARTIFACT LOGGING INTERACTION
        # -------------------------------------------------------------
        if mlflow.active_run():
            print(f"[MLflow] Active tracking context found. Syncing '{report_path}' to server artifacts...")
            mlflow.log_artifact(report_path, artifact_path="xai_reports")
        else:
            print("[INFO] No active MLflow run detected during explanation stage execution. Skipping remote sync.")
            
        return True

    def run(self):
        print("========================================")
        print("      STARTING ADR-NLP MASTER ENGINE     ")
        print("========================================\n")
        
        # Configure the central tracking environment variables globally
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        train_df, val_df, test_df = None, None, None
        model, tokenizer = None, None

        if self.config.get("pipeline", {}).get("run_data", True):
            print("Executing Data Pipeline Stage...")
            train_df, val_df, test_df = run_data_pipeline(self.config)

        # -------------------------------------------------------------
        # TRAINING & MLFLOW SESSION CONTEXT MANAGEMENT
        # -------------------------------------------------------------
        if self.config.get("pipeline", {}).get("run_training", True):
            print("\nExecuting Training Pipeline Stage...")
            # Training pipeline manages its own nested/active run context inside here
            model, tokenizer = run_training_pipeline(
                self.config, train_df, val_df, test_df
            )
        
        if self.config.get("pipeline", {}).get("run_xai", True):
            print("\nExecuting XAI Pipeline Stage...")
            if model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model_path = self.config.get("paths", {}).get("models", "models/adr-nlp-final")
                print(f"Loading saved model checkpoints for XAI from: {model_path}")
                
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception as e:
                    print(f"Critical Error: Could not resolve model binaries for XAI: {e}")
                    print("Skipping XAI verification. Please run the training pipeline first.")
                    return

            # If training ran, an active run context might already persist. 
            # If training was skipped, we spin up a new run exclusively to track the XAI report.
            if not mlflow.active_run():
                print("[MLflow] Initializing a dedicated XAI evaluation run context...")
                with mlflow.start_run(run_name="xai_standalone_evaluation"):
                    self.run_explanation_stage(model, tokenizer, test_df)
            else:
                self.run_explanation_stage(model, tokenizer, test_df)

        print("\n========================================")
        print("      ENGINE MASTER RUN COMPLETED       ")
        print("========================================")


if __name__ == "__main__":
    config = load_config()
    engine = Engine(config)
    engine.run()
import os
import yaml
import pandas as pd
import mlflow  # Integrated for central experiment tracking tracking
from data_pipeline import run_data_pipeline
from training_pipeline import run_training_pipeline
from explainerAI import XAIEngine
from src.config_loader import load_config 


class Engine:
    def __init__(self, config):
        self.config = config

    def run_explanation_stage(self, model, tokenizer, test_df):
        print("\n--- Starting XAI Explanation Stage ---")
        
        text_col = self.config.get("dataset", {}).get("text_column", "review")
        
        if test_df is None:
            split_dir = self.config.get("paths", {}).get("split_dir", "data/processed/split")
            test_path = os.path.join(split_dir, "test.csv")
            print(f"Loading test set split for XAI from: {test_path}")
            test_df = pd.read_csv(test_path)

        xai = XAIEngine(model, tokenizer)

        sample_text = test_df[text_col].iloc[0]
        print(f"Explaining sample from column '{text_col}': {sample_text[:60]}...")

        # target output directory exists before generating reports
        os.makedirs("reports", exist_ok=True)
        report_path = "reports/shap_output.html"
        xai.explain(sample_text, save_path=report_path)
        print("XAI Stage Complete: SHAP values generated and saved.")

        # -------------------------------------------------------------
        # MLFLOW ARTIFACT LOGGING INTERACTION
        # -------------------------------------------------------------
        if mlflow.active_run():
            print(f"[MLflow] Active tracking context found. Syncing '{report_path}' to server artifacts...")
            mlflow.log_artifact(report_path, artifact_path="xai_reports")
        else:
            print("[INFO] No active MLflow run detected during explanation stage execution. Skipping remote sync.")
            
        return True

    def run(self):
        print("========================================")
        print("      STARTING ADR-NLP MASTER ENGINE     ")
        print("========================================\n")
        
        # Configure the central tracking environment variables globally
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        train_df, val_df, test_df = None, None, None
        model, tokenizer = None, None

        if self.config.get("pipeline", {}).get("run_data", True):
            print("Executing Data Pipeline Stage...")
            train_df, val_df, test_df = run_data_pipeline(self.config)

        # -------------------------------------------------------------
        # TRAINING & MLFLOW SESSION CONTEXT MANAGEMENT
        # -------------------------------------------------------------
        if self.config.get("pipeline", {}).get("run_training", True):
            print("\nExecuting Training Pipeline Stage...")
            # Training pipeline manages its own nested/active run context inside here
            model, tokenizer = run_training_pipeline(
                self.config, train_df, val_df, test_df
            )
        
        if self.config.get("pipeline", {}).get("run_xai", True):
            print("\nExecuting XAI Pipeline Stage...")
            if model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model_path = self.config.get("paths", {}).get("models", "models/adr-nlp-final")
                print(f"Loading saved model checkpoints for XAI from: {model_path}")
                
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception as e:
                    print(f"Critical Error: Could not resolve model binaries for XAI: {e}")
                    print("Skipping XAI verification. Please run the training pipeline first.")
                    return

            # If training ran, an active run context might already persist. 
            # If training was skipped, we spin up a new run exclusively to track the XAI report.
            if not mlflow.active_run():
                print("[MLflow] Initializing a dedicated XAI evaluation run context...")
                with mlflow.start_run(run_name="xai_standalone_evaluation"):
                    self.run_explanation_stage(model, tokenizer, test_df)
            else:
                self.run_explanation_stage(model, tokenizer, test_df)

        print("\n========================================")
        print("      ENGINE MASTER RUN COMPLETED       ")
        print("========================================")


if __name__ == "__main__":
    config = load_config()
    engine = Engine(config)
    engine.run()

