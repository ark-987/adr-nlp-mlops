# Project Name
adr-nlp

mermaid
```
flowchart TD
    %% Define External Infrastructure
    subgraph GitHub_Actions [CI/CD: GitHub Actions]
        direction TB
        G1[Trigger: Push to Main] --> G2[Run Tests & Linting]
        G2 --> G3[Run DVC Pipeline Checks]
    end

    %% DVC Data & Training Pipeline (GCP Cloud Core)
    subgraph DVC_Pipeline [Data & Training: DVC on GCP]
        node4[ingest: from GCS] --> node1[data_pipeline]
        node1 --> node7[split]
        node7 --> node8[train: saves to GCS]
        node8 --> node2[explainability]
        node7 --> node2
    end

    %% Experiment Tracking Layer
    subgraph Tracking [Tracking: MLflow Server]
        MFT[MLflow Tracking: Logs Hyperparameters & Metrics]
    end
    node8 -.->|Log Runs| MFT

    %% Deployment Infrastructure (AWS Production Core)
    subgraph Deployment [Deployment: Docker, ECR & FastAPI on AWS]
        node8 -->|CI/CD Sync to S3| S3[(Amazon S3 Bucket)]
        node5[package_api: Docker Build] --> ECR[(Amazon ECR Registry)]
        ECR -->|Pull Docker Image| EC2[Amazon EC2 Instance]
        S3 -->|Download Model Weights| EC2
        
        subgraph Docker_Container [Inside Running Docker Container]
            CORS[CORS Middleware Layer] --> F1[FastAPI Server & Uvicorn]
        end
        EC2 --> Docker_Container
    end

    %% Orchestration & Monitoring
    subgraph Operations [Operations: Monitoring Stack]
        node5 --> node6[setup_monitoring]
        F1 -- Metrics --> M1[Prometheus]
        M1 -- Dashboard --> M2[Grafana]
    end

    %% Connect CI/CD to Pipeline
    G3 --> node4
```

# BioBERT Adverse Drug Reaction (ADR) Classifier 

An end-to-end hybrid-cloud MLOps pipeline designed to train, track, and deploy a clinical sequence classification model (BioBERT) that detects Adverse Drug Reactions from patient reviews. 

The architecture bridges **Google Cloud Platform (GCS)** for early-stage raw data ingestion, preprocessing, and training artifacts with **Amazon Web Services (AWS EC2)** for low-latency production container inference. The pipeline integrates **DVC** for data reproducibility, **MLflow** for experiment tracking, **FastAPI + Streamlit** for serving, **Docker** for container isolation, and a **Prometheus + Grafana** stack for live telemetry monitoring



* **Development/Training & Data Archive Phase**: Raw text data and final trained models are housed inside **Google Cloud Storage (GCS)**. Proceeding with Google free tier and limited RAM on local computer lead to adopting AWS cloud infrastructure for deployment. DVC already logged development in GCP therefore development not rerun in AWS.
* **Production Deployment Serving Phase**: The containerized runtime pulls weights directly from an **Amazon S3** bucket into ECR and SSH into EC2 instance. 


### How to Run a Safe Local Infrastructure Test:
The system features local fallback configurations that bypass the cloud download layers if no AWS keys are passed. To explicitly protect remote production buckets during manual modifications, open `config/pipeline.yaml` and verify local mock destination values:
```yaml
aws:
  bucket_name: "test-dummy-s3-bucket"
  model_s3_path: "models/test_dummy_model.zip"

paths:
  model_dir: "models/adr-nlp-final"
```

##  Project Architecture & Layout

```text
adr-nlp-mlops/
├── .github/
│   └── workflows/
│       └── cicd.yaml             # CI/CD: Multi-container build, ECR tagging, and lint check
├── config/                       # Central configuration schemas
│   ├── data.yaml                 # Dataset boundary specifications
│   ├── pipeline.yaml             # Main AWS, GCP, MLflow, and local path variables
│   └── prometheus.yaml           # Telemetry: Metrics scraping targets pointing to api:8000
├── data/                         # Local storage volumes (Ignored by Git)
│   ├── dummy/                    # 20-review safe validation seed files
│   ├── processed/                # Cleansed and partitioned data arrays
│   ├── raw/                      # Raw unescaped target reviews ingested from GCS
│   └── mock_aws_s3_bucket/       # Offline verification mock model weights fallback cache
├── docs/                         # Automated Sphinx codebase documentation
├── gx/                           # Great Expectations Data Quality Layer
│   ├── checkpoints/              # Schema testing validation manifests
│   ├── expectations/             # JSON asset data property assertions
│   └── great_expectations.yml    # Framework global configuration file
├── logs/
│   └── metrics.json                # Evaluation stats monitored by DVC
├── mlruns/                         # Local experiment database folder
├── models/
│   └── adr-nlp-final/            # Destination for unpacked BioBERT model shards at API boot
├── notebooks/                    # Experimental research Jupyter notebooks
├── references/                   # Explanatory operational training manuals
├── reports/                      # Visual output charts and static figures
├── src/                          # System core python package modules
│   ├── __init__.py
│   ├── api.py                    # FastAPI prediction serving engine (Instrumented with Prometheus)
│   ├── build_features.py         # Feature engineering orchestration logic
│   ├── cleaning_agent.py         # Text cleaning and normalisation algorithms
│   ├── config_loader.py          # Absolute path resolution config loader for cross-environment safety
│   ├── data_pipeline.py          # Sequence execution controller
│   ├── explainerAI.py            # SHAP model explainability (XAI) backend
│   ├── gcs_utils.py              # GCP cloud storage data ingestion utilities
│   ├── s3_utils.py               # AWS cloud storage production model streaming/download utilities
│   ├── ge_expectations.py        # Great Expectations suite generator
│   ├── ge_validator.py           # Verification runtime checkpoint runner
│   ├── ingest.py                 # Remote dataset ingestion target gateway (GCS focused)
│   ├── orchestrator.py           # Pipeline master structural workflow manager
│   ├── split.py                  # Partitions clean arrays reproducibly
│   └── training_pipeline.py      # RAM-optimized training loop controller (Saves to GCS)
├── tests/                        # Automated code verification suites
│   ├── __init__.py
│   ├── test_api.py               # Unit assertions for mock FastAPI responses
│   ├── test_cleaning_agent.py    # Checks string cleaning normalization 
│   └── test_trainer.py           # Validates model structural initialization
├── app.py                        # Streamlit browser dashboard UI user interface layer
├── Dockerfile                    # Isolated production Backend FastAPI server build steps
├── Dockerfile.frontend           # Isolated production Frontend Streamlit UI build steps
├── docker-compose.yaml           # 4-Container local ecosystem coordinator (API, UI, Prom, Grafana)
├── dvc.lock                      # Frozen state snapshot of executed DVC pipelines
├── dvc.yaml                      # Reproducible pipeline step configurations
├── main.py                       # Project entry point execution script
├── Makefile                      # Standardized terminal command shortcuts
├── params.yaml                   # Central model parameter definition track
├── requirements.txt              # Production image runtime software dependencies list
├── requirements_dev.txt          # Development, GCS data ingestion, and training dependencies list
├── setup.py                      # Local editable package installation anchor
├── test_environment.py           # Baseline hardware compatibility checking script
└── tox.ini                       # Multi-environment automation checker
```

---

##  Getting Started (Local Runtime Multi-Container Boot)

### 1. Prerequisites
* Python 3.10+ installed.
* Docker Desktop installed, configured with WSL2 background virtualization.
* Your laptop's active AWS credentials configured (`~/.aws` containing valid keys) to stream production weights.

### 2. Dependency Separation
Code requirements split between development and deployment:
* **Development/Training Phase**: Install `requirements_dev.txt` (Installs `google-cloud-storage`, `dvc[gcs,s3]`, PyTorch, and training frameworks).
* **Production Serving Phase**: The Docker images use `requirements.txt` to eliminate heavy multi-cloud footprint overhead.

```bash
# Clone the repository
git clone <https://github.com/ark-987/adr-nlp-mlops>
cd adr-nlp-mlops

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Mac/Linux: source venv/bin/activate

# Install Full Development and Training Stack (Includes GCP + AWS tools)
pip install -r requirements_dev.txt
```

### 3. Orchestrating the Monitoring Stack Locally
To compile your custom images, download utility databases, and wire the local networks together, run:
```bash
docker compose up -d --build
```
Once the health check switches to healthy inside `docker compose ps`, access your services across your terminal network ports:
* **Streamlit UI Application Interface**: `http://localhost:8501`
* **FastAPI Swagger API Documentation**: `http://localhost:8000/docs`
* **Prometheus Metrics Scraper**: `http://localhost:9090`
* **Grafana Visual Analytics Canvas**: `http://localhost:3000` (Login: `admin` / `admin`)

---

##  Pipeline Elements

### Phase 1: Data Ingestion & Quality Gates (GCP Core)
* **`src/ingest.py`**: pulls raw `csv` training datasets down from your GCS data lake bucket into local workspace.
* **`src/ge_validator.py`**: Executes **Great Expectations** data validation checks, verifying strings and schemas directly inside your ingestion stage before deep learning steps execute.

### Phase 2: Serving & Production Monitoring (AWS Core)


