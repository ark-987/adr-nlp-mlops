import pandas as pd
import mlflow.transformers
import logging

def main(input_filepath, output_filepath):
    """
    Loads the registered model and runs inference on a CSV of new reviews.
    """
    logger = logging.getLogger(__name__)
    
    # 1. Load the model from the local registry/GCS
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/models/biobert_adr_classifier/" 
    logger.info(f"Loading model: {model_uri}")
    pipe = mlflow.transformers.load_model(model_uri)

    # 2. Load new data
    df = pd.read_csv(input_filepath)
    
    # 3. Predict
    logger.info(f"Predicting on {len(df)} samples...")
    # Map the pipeline over the text column
    results = df['review'].apply(lambda x: pipe(x)[0])
    
    # 4. Format and Save
    df['prediction'] = [res['label'] for res in results]
    df['confidence'] = [res['score'] for res in results]
    
    df.to_csv(output_filepath, index=False)
    logger.info(f"Predictions saved to {output_filepath}")

if __name__ == "__main__":
    # Example usage
    main("data/raw/new_batch.csv", "data/processed/predictions.csv")
