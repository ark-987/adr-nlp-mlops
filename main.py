# main.py (at the project root)
from src.engine.orchestrator import Engine 
from src.utils.config_loader import load_config

def main():
    print("--- ADR-NLP Entry Point ---")
    
    # 1. Load the config using central loader
    # This will automatically find configs/pipeline.yaml
    config = load_config()
    
    # 2. Initialize the Engine 
    engine = Engine(config)
    
    # 3. Run the orchestration (Data -> Training)
    engine.run()

if __name__ == "__main__":
    main()
