import logging
from src.utils.config_loader import load_config
# Importing the actual function name from your root-level logic
from src.pipelines.training_pipeline import run_training_pipeline 

def main():
    """
    Entry point for training the model. 
    Standard Cookiecutter logic: Reads data from processed and saves to models.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline...")
    
    try:
        # 1. Load the centralized YAML configuration
        config = load_config()
        
        # 2. Run the pipeline
        # Passing None for dataframes tells your run_training_pipeline 
        # to pull the CSVs from GCS/local paths as defined in your code.
        model, tokenizer = run_training_pipeline(
            config=config, 
            train_df=None, 
            val_df=None, 
            test_df=None
        )
        
        logger.info("Training completed successfully and model objects returned.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Raising the error is important so DVC knows the stage failed
        raise e

if __name__ == "__main__":
    main()
