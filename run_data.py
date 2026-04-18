from src.utils.config_loader import load_config
from src.pipelines.data_pipeline import run_data_pipeline

config = load_config()
run_data_pipeline(config)