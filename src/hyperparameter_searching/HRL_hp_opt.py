import warnings

warnings.filterwarnings("ignore")

from preprocess.preprocessor import get_df
from hyperparameter_searching.HRL_optuna_utils import HyperparamsOptHRL

from config_training import TrainSettings

settings = TrainSettings()

TRAIN_START_DATE = settings.TRAIN_START_DATE
TRAIN_END_DATE = settings.TRAIN_END_DATE
VAL_START_DATE = settings.VAL_START_DATE
VAL_END_DATE = settings.VAL_END_DATE

INDICATORS = settings.INDICATORS

N_TRIALS = 40

df_train = get_df(
    TRAIN_START_DATE, TRAIN_END_DATE, "../src/preprocess/tickers/ticker_lists.json"
)
df_val = get_df(
    VAL_START_DATE, VAL_END_DATE, "../src/preprocess/tickers/ticker_lists.json"
)


hrl_opt = HyperparamsOptHRL(
    df_train=df_train, df_test=df_val, indicators=INDICATORS, n_trials=N_TRIALS
)
hrl_opt.run_opt()
