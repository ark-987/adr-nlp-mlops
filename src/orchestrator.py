import os
import yaml
import pandas as pd
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
        xai.explain(sample_text, save_path="reports/shap_output.html")
        
        print("XAI Stage Complete: SHAP values generated and saved.")
        return True

    def run(self):
        print("========================================")
        print("      STARTING ADR-NLP MASTER ENGINE     ")
        print("========================================\n")
        
        train_df, val_df, test_df = None, None, None
        model, tokenizer = None, None

        if self.config.get("pipeline", {}).get("run_data", True):
            print("Executing Data Pipeline Stage...")
            train_df, val_df, test_df = run_data_pipeline(self.config)

        if self.config.get("pipeline", {}).get("run_training", True):
            print("\nExecuting Training Pipeline Stage...")
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

            self.run_explanation_stage(model, tokenizer, test_df)

        print("\n========================================")
        print("      ENGINE MASTER RUN COMPLETED       ")
        print("========================================")


if __name__ == "__main__":
    config = load_config()
    engine = Engine(config)
    engine.run()
