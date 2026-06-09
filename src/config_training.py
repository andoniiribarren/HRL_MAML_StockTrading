from pathlib import Path

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(name: str) -> dict:
    with open(_CONFIGS_DIR / name, "r") as f:
        return yaml.safe_load(f)


class TrainSettings:
    def __init__(self):
        defaults = _load_yaml("defaults.yaml")
        hp = _load_yaml("best_hyperparams.yaml")

        # Directories
        dirs = defaults["directories"]
        self.DATA_SAVE_DIR = dirs["data_save"]
        self.TRAINED_MODEL_DIR = dirs["trained_models"]
        self.TENSORBOARD_LOG_DIR = dirs["tensorboard_log"]
        self.RESULTS_DIR = dirs["results"]

        # Date ranges
        dates = defaults["dates"]
        self.TRAIN_START_DATE = dates["train_start"]
        self.TRAIN_END_DATE = dates["train_end"]
        self.VAL_START_DATE = dates["val_start"]
        self.VAL_END_DATE = dates["val_end"]
        self.TEST23_START_DATE = dates["test23_start"]
        self.TEST23_END_DATE = dates["test23_end"]
        self.TEST24_START_DATE = dates["test24_start"]
        self.TEST24_END_DATE = dates["test24_end"]

        # Technical indicators
        self.INDICATORS = defaults["indicators"]

        # Best hyperparameters
        self.best_hiperparams_RL = hp["rl_baseline"]
        self.best_hiperparams_HRL = hp["hrl"]
        self.best_hiperparams_HRL_worker_update = hp["hrl_worker_update"]
