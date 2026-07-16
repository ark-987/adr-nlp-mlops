FROM python:3.10-slim


# ==========================================================
# Runtime configuration
# ==========================================================

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app



WORKDIR /app



# ==========================================================
# System dependencies
# ==========================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*



# ==========================================================
# Install Python dependencies
# ==========================================================

COPY requirements.txt .


RUN python -m pip install --upgrade pip



# PyTorch CPU build for BioBERT inference

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu



RUN pip install --no-cache-dir \
    -r requirements.txt




# ==========================================================
# Application files
# ==========================================================


COPY src/ ./src/

COPY config/ ./config/



# Runtime model location
# This is where S3 extraction will place the model

RUN mkdir -p /app/models/production_live_adr_nlp



# ==========================================================
# FastAPI port
# ==========================================================

EXPOSE 8000




# ==========================================================
# Application startup
# ==========================================================

CMD [
    "uvicorn",
    "src.api:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]

