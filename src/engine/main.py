from src.utils.config_loader import load_config
from src.pipelines.data_pipeline import run_data_pipeline


class Engine:

    def __init__(self, config):
        self.config = config

    def run(self):
        print("DEBUG CONFIG TYPE:", type(self.config))
        print("DEBUG CONFIG VALUE:", self.config)
        print("Starting pipeline")
        train_df, val_df, test_df = None, None, None

        # -------------------------
        # DATA PIPELINE
        # -------------------------
        if self.config.get("pipeline", {}).get("run_data", True):
            print("Running data pipeline...")
            train_df, val_df, test_df = run_data_pipeline(self.config)

        # -------------------------
        # TRAINING PIPELINE
        # -------------------------
        if self.config.get("pipeline", {}).get("run_training", True):
            from src.pipelines.training_pipeline import run_training_pipeline
            print("Running training pipeline...")

            # Safety check
            if train_df is None:
                raise ValueError("Training requires data. Run data pipeline first.")

            run_training_pipeline(
                self.config,
                train_df,
                val_df,
                test_df
            )

        print("Pipeline complete")


if __name__ == "__main__":
    config = load_config()
    engine = Engine(config)
    engine.run()