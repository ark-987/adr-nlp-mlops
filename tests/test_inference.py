import mlflow.transformers
import torch

# 1. Point to your local tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Load the model directly from the Registry
# Use the name you registered in the training script
#model_uri = "models:models/biobert_adr_classifier/" 
#logger.info(f"Loading model from {model_uri}...")

#change to local path for testing
# Instead of "models:/..." use the local path DVC just created
model_path = "./models/adr-nlp-final" 


# This loads both the model and the tokenizer automatically
pipe = mlflow.transformers.load_model(model_path)

# 3. Define test cases
test_reviews = [
    "I took this medication and developed a severe skin rash within an hour.", # Expected: ADR (1)
    "The pills arrived on time and the packaging was great."                  # Expected: No ADR (0)
]

# 4. Run the test
print("\n--- Inference Results ---")
for review in test_reviews:
    result = pipe(review)
    print(f"Review: {review}")
    print(f"Result: {result}\n")
