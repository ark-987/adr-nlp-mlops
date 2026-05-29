import os
import sys

# Dynamically inject the root path so nested imports resolve cleanly [2, 3]
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.orchestrator import Engine 
from src.config_loader import load_config


def main():
    print("========================================")
    print("      ADR-NLP SYSTEM ENTRY POINT        ")
    print("========================================\n")
    
    # 1. Load the central yaml schema from config/pipeline.yaml
    config = load_config()
    
    # 2. Initialize the Master Orchestrator Engine 
    engine = Engine(config)
    
    # 3. Run the orchestration pipeline (Data Ingest -> Train -> XAI)
    engine.run()


if __name__ == "__main__":
    main()

