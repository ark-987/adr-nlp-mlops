adr-nlp
==============================

ADR classification using NLP of noisy dataset with transformer driven by an engine within MLOPs workflow

Project Organization
------------

    ├── LICENSE
    ├── Makefile           
    ├── README.md          
    ├── data
    │   ├── external       
    │   ├── interim        
    │   ├── processed      
    │   └── raw            
    │
    ├── docs               <- A default Sphinx document tree; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- data exploration in kaggle and colab with GPU accelerator for training, building and evaluating ADR-prediction model
    │
    ├── references         
    │
    ├── reports            
    │   └── figures       
    │
    ├── requirements.txt   < generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- src is a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features                   <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py      <-ADR labeling by enrichment model
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>




MLOPs ARCHITECTURE OF ADR NLP ORCHESTRATED BY AN ENGINE WITH AN AI AGENT INTEGRATION 

                             ┌────────────────────────────┐
                             │        CONFIG (YAML)       │
                             │  - pipeline.yaml           │
                             │  - agent settings          │
                             │  - paths                   │
                             └─────────────┬──────────────┘
                                           │
                                           ▼
                             ┌────────────────────────────┐
                             │        ENGINE              │  
                             │  (Custom Orchestrator)     │
                             └─────────────┬──────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                  │
        ▼                                  ▼                                  ▼
┌──────────────┐                  ┌──────────────┐                  ┌──────────────────────┐
│  Ingestion   │                  │  Validation  │                  │     Agent Layer      │
│ (Kaggle API) │                  │ (Great Exp.) │                  │   (LLM Cleaning)     │
└──────┬───────┘                  └──────┬───────┘                  └─────────┬────────────┘
       │                                 │                                   │
       ▼                                 ▼                                   ▼
                       ┌────────────────────────────────────────────────────────┐
                       │                AGENT CONTROL ZONE                       │
                       │                                                        │
                       │   ┌──────────────┐      ┌──────────────┐               │
                       │   │   CACHE      │◄────►│ CleaningAgent│               │
                       │   │ (JSON/Redis) │      │ (LangChain)  │               │
                       │   └──────┬───────┘      └──────┬───────┘               │
                       │          │                      │                       │
                       │          ▼                      ▼                       │
                       │   ┌──────────────────────────────────────┐              │
                       │   │ Logging / Tracking Layer             │              │
                       │   │ (MLflow + structured logs)           │              │
                       │   └──────────────────────────────────────┘              │
                       └──────────────────────┬─────────────────────────────────┘
                                              │
                                              ▼
                    ┌────────────────────────────────────────────┐
                    │         CLEANED / AUGMENTED DATA           │
                    └──────────────────────────┬─────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ Preprocessing    │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Training (BERT)  │
                                    │ Hugging Face     │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Evaluation + XAI │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ MLflow Registry  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Deployment       │
                                    │ (FastAPI + Docker)
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Monitoring       │
                                    │ Prometheus/Grafana
                                    └──────────────────┘


──────────────────────────────────────────────────────────────────────────────
🚀 CI/CD LAYER (triggered on git push / PR)
──────────────────────────────────────────────────────────────────────────────

        Developer Push → GitHub Repo
                         │
                         ▼
        ┌──────────────────────────────────────────────┐
        │:contentReference[oaicite:1]{index=1} CI/CD Pipeline │
        └──────────────────────────────────────────────┘
                         │
     ┌───────────────────┼───────────────────────────────┐
     │                   │                               │
     ▼                   ▼                               ▼
┌──────────────┐   ┌──────────────┐               ┌──────────────────┐
│ Linting      │   │ Unit Tests   │               │ Data Validation  │
│ (flake8)     │   │ (pytest)     │               │ (Great Exp.)     │
└──────┬───────┘   └──────┬───────┘               └────────┬─────────┘
       │                  │                                │
       └──────────────────┴────────────────────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Build Docker     │
                     │ Image            │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Push to Registry │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Deploy API       │
                     │ (FastAPI)        │
                     └──────────────────┘


