# Use a lightweight, official Python runtime as a parent image
FROM python:3.10-slim

# Set environment system configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="."

# Set the working directory inside the container box
WORKDIR /app

# Install system dependencies required for compiling certain Python binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker caching layers
COPY requirements.txt .

# Install dependencies directly into the container instance
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the configuration directories and core source modules
COPY config/ ./config/
COPY src/ ./src/

# Expose port 8000 for network routing traffic gates
EXPOSE 8000

# Boot instruction: Launches the application server utilizing Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
