import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_fastapi_instrumentator import Instrumentator

# Local core pipeline imports
from src.config_loader import load_config
from src.s3_utils import download_and_extract_model_from_s3 

# Define global placeholder hooks to freeze BioBERT layers inside container RAM
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes automatically EXACTLY ONCE when the server container turns on."""
    global model, tokenizer
    
    try:
        # Dynamically map cloud schemas straight from your pipeline.yaml variables
        config = load_config()
        bucket_name = config["aws"]["bucket_name"]         
        model_blob_name = config["aws"]["model_s3_path"]   
        local_extract_dir = config["paths"]["model_dir"]   
    except KeyError as e:
        print(f"[CRITICAL] Config is missing a required AWS operational key: {e}")
        raise RuntimeError("System configuration structural parsing failed.")

    # Call your AWS Downloader
    success = download_and_extract_model_from_s3(
        bucket_name=bucket_name,
        source_s3_key=model_blob_name,
        extract_to_dir=local_extract_dir
    )
    
    if not success:
        print("[CRITICAL] BioBERT assets could not be retrieved from AWS. Server halting boot.")
        raise RuntimeError("Model download gate failure.")
        
    print(f"[BOOT] Initializing BioBERT weights into server memory from: {local_extract_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_extract_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_extract_dir)
    print("[BOOT] Server application memory securely initialized! Accepting network traffic.")
    yield

# Instantiate the primary FastAPI engine with the runtime initialization lifecycles
app = FastAPI(lifespan=lifespan)

# BROWSER SAFETY: Enable Cross-Origin Handshaking with your Streamlit UI components
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INPUT GUARDRAIL: Strict incoming type and token bounding schema validation using Pydantic
class ReviewInput(BaseModel):
    review: str = Field(..., min_length=5, max_length=1000)

# =========================================================================
# SYSTEM NETWORK ROUTES (ENDPOINTS)
# =========================================================================

@app.get("/health")
async def liveness_probe():
    """Liveness probe checkpoint for Docker Compose health checking metrics."""
    return {"status": "healthy"}

@app.post("/predict")
async def predict_adr(data: ReviewInput):
    """Processes incoming data streams, executes inference matrices, returns classifications."""
    global model, tokenizer
    
    # Baseline protection for pure whitespace extraction strings
    if data.review.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Review payload cannot consist purely of space strings."
        )
    
    try:
        # Leverage your Hugging Face dependencies to tokenise strings into raw tensors
        inputs = tokenizer(data.review, return_tensors="pt", truncation=True, padding=True)
        
        # Execute the forward pass calculations across active neural networks in memory
        outputs = model(**inputs)
        
        # Mapping mock responses for architecture confirmation 
        # (Replace with your direct tensor argmax logic once hardware layers are validated)
        prediction_class_id = 1 
        class_probabilities = [0.15, 0.85]
        
        return {
            "prediction_class_id": prediction_class_id,
            "class_probabilities": class_probabilities
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Inference Pipeline Process Failure: {str(e)}"
        )

# =========================================================================
# TELEMETRY REGISTRATION (Must sit at bottom to intercept registered endpoints)
# =========================================================================
Instrumentator().instrument(app).expose(app)


