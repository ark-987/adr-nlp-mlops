# src/server.py
import os
import zipfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import your unified config loader directly from your codebase
from src.config_loader import load_config
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="BioBERT ADR Classifier API", 
    description="Production serving layer for evaluating Adverse Drug Reactions."
)
#ADD THIS HOOK LAYER!
# This automatically creates and updates the /metrics gateway for Prometheus
Instrumentator().instrument(app).expose(app)
# Global memory hooks for the model assets
model = None
tokenizer = None
config = None

class PredictionRequest(BaseModel):
    review: str

@app.on_event("startup")
def startup_load_pipeline():
    """Executes automatically once when the FastAPI server initializes."""
    global model, tokenizer, config
    print("[BOOT] Loading unified pipeline configurations...")
    config = load_config()

    bucket_name = config["gcp"]["bucket_name"]
    model_zip_file = config["gcp"]["model_gcs_file"]
    local_extract_dir = config["paths"]["models"]
    local_zip_path = os.path.join(local_extract_dir, "downloaded_model.zip")

    # Ensure local directory footprint path exists
    os.makedirs(local_extract_dir, exist_ok=True)

    # 1. Download production binaries from your GCS Bucket
    print(f"[BOOT] Fetching production model 'gs://{bucket_name}/{model_zip_file}'...")
    try:
        from google.cloud import storage
        client = storage.Client(project=config["gcp"]["project_id"])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_zip_file)
        
        # Stream the zipped asset directly to local temp directory path
        blob.download_to_filename(local_zip_path)
        print("[BOOT] Download complete. Extracting model shards...")
        
        # 2. Decompress the production model zip package
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_extract_dir)
            
        # Clean up the raw zip archive instantly to preserve your local disk space
        os.remove(local_zip_path)
        
    except Exception as e:
        print(f"[FATAL INITIALIZATION ERROR] Cloud connection failed: {e}")
        raise RuntimeError(f"Could not load remote cloud production assets: {e}")

    # 3. Load weights out of local directory space straight into active application memory
    print(f"[BOOT] Loading weight arrays into application RAM...")
    tokenizer = AutoTokenizer.from_pretrained(local_extract_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_extract_dir)
    
    # Enforce evaluation context mode (turns off dropout, freeze gradients)
    model.eval()
    print("[BOOT] Serving engine successfully activated. Ready for text streaming requests.")

@app.get("/health")
def health_check():
    """Liveness probe monitoring endpoint for Docker/Kubernetes routing hooks."""
    if model is not None and tokenizer is not None:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Model initialization incomplete")

@app.post("/predict")
def predict_adverse_reaction(request: PredictionRequest):
    """Processes unformatted medical text entries to evaluate structural ADR metrics."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Serving model parameters are offline")

    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review entry text string cannot be empty")

    inputs = tokenizer(
        request.review, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()

    return {
        "prediction_class_id": predicted_class_id,
        "class_probabilities": probabilities
    }

if __name__ == "__main__":
    import uvicorn
    # Forces Uvicorn to look at the module path relative to your repository root folder
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
