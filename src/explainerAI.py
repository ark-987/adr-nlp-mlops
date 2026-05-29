import os
import shap
import transformers

class XAIEngine:
    def __init__(self, model, tokenizer):
        # Explicitly setting top_k=None handles class arrays cleanly
        self.pipeline = transformers.pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=-1, # CPU avoids CUDA instability during SHAP generation
            top_k=None 
        )
        # Partition explainer resolves text boundaries naturally
        self.explainer = shap.Explainer(self.pipeline)

    def explain(self, text, save_path=None):
        """
        Generates SHAP values and saves an interactive HTML visualization.
        """
        shap_values = self.explainer([text])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Capture the interactive HTML structure safely
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(shap.plots.text(shap_values, display=False))
            print(f"XAI Interactive Report generated at: {save_path}")

        return shap_values


