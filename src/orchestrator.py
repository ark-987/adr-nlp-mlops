import os
import yaml
import pandas as pd

# Standardised local imports from your updated project structure
from data_pipeline import run_data_pipeline
from training_pipeline import run_training_pipeline
from explainerAI import XAIEngine


class Engine:
    def __init__(self, config):
        self.config = config

    def run_explanation_stage(self, model, tokenizer, test_df):
        print("\n--- Starting XAI Explanation Stage ---")
        
        text_col = self.config.get("dataset", {}).get("text_column", "review")
        
        if test_df is None:
            # Aligned exactly to the split path generated inside split.py
            split_dir = self.config.get("paths", {}).get("split_dir", "data/processed/split")
            test_path = os.path.join(split_dir, "test.csv")
            print(f"Loading test set split for XAI from: {test_path}")
            test_df = pd.read_csv(test_path)

        # Initialize the XAI Engine
        xai = XAIEngine(model, tokenizer)

        # Grab a sample from the specific column your trainer uses
        sample_text = test_df[text_col].iloc[0]
        print(f"Explaining sample from column '{text_col}': {sample_text[:60]}...")

        # Generates and outputs interactive SHAP reports
        xai.explain(sample_text, save_path="reports/shap_output.html")
        
        print("XAI Stage Complete: SHAP values generated and saved.")
        return True

    def run(self):
        print("========================================")
        print("      STARTING ADR-NLP MASTER ENGINE     ")
        print("========================================\n")
        
        train_df, val_df, test_df = None, None, None
        model, tokenizer = None, None

        # 1. DATA PIPELINE (Ingestion -> Cleaning -> Validation -> Split -> GCS)
        if self.config.get("pipeline", {}).get("run_data", True):
            print("Executing Data Pipeline Stage...")
            train_df, val_df, test_df = run_data_pipeline(self.config)

        # 2. TRAINING PIPELINE (HuggingFace Datasets -> NER Features -> BioBERT)
        if self.config.get("pipeline", {}).get("run_training", True):
            print("\nExecuting Training Pipeline Stage...")
            model, tokenizer = run_training_pipeline(
                self.config, train_df, val_df, test_df
            )
        
        # 3. XAI EXPLANATION PIPELINE (SHAP Visualizations)
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

            self.run_explanation_stage(model, tokenizer, test_df)

        print("\n========================================")
        print("      ENGINE RUN COMPLETED ALIGNED      ")
        print("========================================")


if __name__ == "__main__":
    # Local fallback safe configuration parser
    try:
        with open("pipeline.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: pipeline.yaml configuration file missing in root directory.")
        exit(1)

    engine = Engine(config)
    engine.run()
