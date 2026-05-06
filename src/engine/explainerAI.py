# src/engine/explainerAI.py
import shap
import transformers
import os

class XAIEngine:
    def __init__(self, model, tokenizer):
        # Explicitly setting return_all_scores=True is crucial for multi-class SHAP analysis
        self.pipeline = transformers.pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=-1, # CPU avoids CUDA-related stability issues during SHAP generation
            top_k=None # Ensures we get probabilities for all labels
        )
        # Passing a pipeline directly allows SHAP to auto-select the 'Partition' explainer
        self.explainer = shap.Explainer(self.pipeline)

    def explain(self, text, save_path=None):
        """
        Generates SHAP values and optionally saves an interactive HTML report.
        """
        shap_values = self.explainer([text])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # shap.plots.text produces an interactive HTML visualization
            # To save it, we capture the HTML representation
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(shap.plots.text(shap_values, display=False))
            print(f"XAI Report saved to: {save_path}")

        return shap_values
