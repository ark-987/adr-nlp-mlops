from src.utils.config_loader import load_config
from src.pipelines.data_pipeline import run_data_pipeline
from src.pipelines.training_pipeline import run_training_pipeline # Ensure path is correct
from src.engine.explainerAI import XAIEngine
import pandas as pd

class Engine:
    def __init__(self, config):
        self.config = config

    def run_explanation_stage(self, model, tokenizer, test_df):
        print("\n--- Starting XAI Explanation Stage ---")
        
        # Determine the correct text column from your config
        text_col = self.config.get("dataset", {}).get("text_column", "review")
        
        if test_df is None:
            test_path = self.config.get("data", {}).get("test_path", "data/processed/test.csv")
            test_df = pd.read_csv(test_path)

        # Initialize the XAI Engine
        xai = XAIEngine(model, tokenizer)

        # Grab a sample from the specific column your trainer uses
        sample_text = test_df[text_col].iloc[0]
        print(f"Explaining sample from column '{text_col}': {sample_text[:60]}...")

        shap_values = xai.explain(sample_text)
        
        print("XAI Stage Complete: SHAP values generated.")
        return shap_values

    def run(self):
        print("Starting ADR-NLP Pipeline")
        train_df, val_df, test_df = None, None, None
        model, tokenizer = None, None

        # 1. DATA PIPELINE
        if self.config.get("pipeline", {}).get("run_data", True):
            print("Running data pipeline...")
            train_df, val_df, test_df = run_data_pipeline(self.config)

        # 2. TRAINING PIPELINE
        if self.config.get("pipeline", {}).get("run_training", True):
            print("Running training pipeline...")
            # This now returns the objects needed for XAI
            model, tokenizer = run_training_pipeline(
                self.config, train_df, val_df, test_df
            )
        
        # 3. XAI EXPLANATION PIPELINE
        if self.config.get("pipeline", {}).get("run_xai", True):
            if model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model_path = self.config.get("paths", {}).get("models", "models/adr-nlp-final")
                print(f"Loading saved model for XAI from: {model_path}")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.run_explanation_stage(model, tokenizer, test_df)

        print("Pipeline execution finished.")

if __name__ == "__main__":
    config = load_config()
    engine = Engine(config)
    engine.run()
