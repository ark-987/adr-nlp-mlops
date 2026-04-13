adr-nlp
==============================

ADR classification using NLP of noisy dataset with transformer driven by an engine within MLOPs workflow

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
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




ARCHITECTURE OF ADR NOP ORCHESTRATED BY AN ENGINE WITH AI AGENT ONTEGRATED plus MLOPs

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
        │          :contentReference[oaicite:1]{index=1} CI/CD Pipeline │
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


