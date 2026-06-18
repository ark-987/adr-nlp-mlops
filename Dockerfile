FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="."

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \     
    build-essential \     
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip && \
# 2. Force install the lightweight CPU torch variant first
    pip install --no-cache-dir torch --index-url https://pytorch.org && \
# 3. Install the rest of the packages from PyPI sequentially to save memory
    pip install --no-cache-dir --no-compile --max-workers=1 -r requirements.txt

COPY config/ ./config/
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

