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
        self.TEST_START_DATE = "2022-01-01"
        self.TEST_END_DATE = "2023-01-01"

        self.INDICATORS = ["macd", "rsi_30", "cci_30"]

        # Model Parameters
