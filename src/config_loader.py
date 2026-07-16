import os
from pathlib import Path

import yaml


def load_config():
    """
    Load the application configuration from config/pipeline.yaml.

    This works consistently whether the application is executed:
      - locally
      - inside Docker
      - via Uvicorn
      - during GitHub Actions tests
    """

    # ----------------------------------------------------------
    # Project root
    # src/config_loader.py -> src -> project root
    # ----------------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    # ----------------------------------------------------------
    # Candidate configuration locations
    # ----------------------------------------------------------

    candidate_paths = [

        project_root / "config" / "pipeline.yaml",

        project_root / "pipeline.yaml",

    ]

    # ----------------------------------------------------------
    # Locate configuration file
    # ----------------------------------------------------------

    config_path = None

    for path in candidate_paths:

        if path.exists():

            config_path = path

            break

    if config_path is None:

        searched = "\n".join(str(p) for p in candidate_paths)

        raise FileNotFoundError(

            f"pipeline.yaml not found.\n\nSearched:\n{searched}"

        )

    # ----------------------------------------------------------
    # Parse YAML
    # ----------------------------------------------------------

    with open(config_path, "r", encoding="utf-8") as f:

        config = yaml.safe_load(f)

    if config is None:

        raise ValueError(

            f"Configuration file is empty: {config_path}"

        )

    return config