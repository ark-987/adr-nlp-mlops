import os
import yaml

def load_config():
    """
    Loads pipeline.yaml using absolute path resolution to guarantee
    successful lookups under local testing, Uvicorn, or Docker run contexts.
    """
    # 1. Determine the absolute directory path of the root project workspace
    # __file__ is src/config_loader.py -> parent is src/ -> grandparent is root/
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, ".."))

    # 2. Map direct absolute paths to your config directory positions
    config_path = os.path.join(project_root, "config", "pipeline.yaml")
    fallback_path = os.path.join(project_root, "pipeline.yaml")

    # 3. Choose the active file pathway location
    final_path = config_path if os.path.exists(config_path) else fallback_path

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Configuration file not found via absolute mapping at: {final_path}")

    # 4. Parse your flat YAML structure
    with open(final_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config





