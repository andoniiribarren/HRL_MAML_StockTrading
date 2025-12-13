class TrainSettings:
    def __init__(self):

        # directory
        self.DATA_SAVE_DIR = "datasets"
        self.TRAINED_MODEL_DIR = "trained_models"
        self.TENSORBOARD_LOG_DIR = "tensorboard_log"
        self.RESULTS_DIR = "results"

        # date format: '%Y-%m-%d'
        self.TRAIN_START_DATE = "2018-01-01"
        self.TRAIN_END_DATE = "2022-01-01"
        self.VAL_START_DATE = "2022-01-01"
        self.VAL_END_DATE = "2023-01-01"

        self.TEST23_START_DATE = "2023-01-01"
        self.TEST23_END_DATE = "2024-01-01"
        self.TEST24_START_DATE = "2024-01-01"
        self.TEST24_END_DATE = "2025-01-01"

        self.INDICATORS = ["macd", "rsi_30", "cci_30"]

        # Model Parameters
        self.best_hiperparams_RL = {
            "gamma": 0.9982572049354244,
            "max_grad_norm": 1.004287538508153,
            "n_steps": 256,
            "learning_rate": 0.0002750235733375285,
            "ent_coef": 6.465134048701665e-05,
        }

        self.best_hiperparams_HRL = {
            "lr_actor_M": 0.00020620848037307665,
            "lr_critic_M": 0.0003374572067862167,
            "gamma_M": 0.9927862051428868,
            "update_timestep": 256,
            "gamma_W": 0.9953330246893773,
            "lr_W": 0.0005967276028261235,
            "buffer_size": 100000,
            "batch_size": 256,
        }
