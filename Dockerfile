FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="."

# Step 1: Copy only requirements to leverage Docker cache
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \     
    build-essential \     
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Step 2: Upgrade pip (cached unless base image changes)
RUN pip install --no-cache-dir --upgrade pip

# Step 3: Install heavy CPU torch (cached unless this specific line changes)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install remaining packages (invalidated only if requirements.txt changes)
RUN pip install --no-cache-dir --no-compile --max-workers=1 -r requirements.txt


COPY config/ ./config/
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]


