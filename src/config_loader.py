import os
import yaml

def load_config():
    """Locates and loads the pipeline.yaml file from the config directory securely."""
    # Calculates path dynamically: go up one level from src/ to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Critical Error: Configuration schema missing at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # AUTOMATED MAPPING: Safely reconstructs the 'training' dictionary if flattened for DVC
    if "training" not in config or config["training"] is None:
        print("[CONFIG LOADER] Reconstructing training dictionary from flat DVC keys...")
        config["training"] = {
            "model_name": config.get("training_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
            "epochs": config.get("training_epochs", 1),
            "batch_size": config.get("training_batch_size", 2)
        }
        
    return config


