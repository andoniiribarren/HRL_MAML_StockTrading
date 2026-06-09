import warnings

warnings.filterwarnings("ignore")

from src.preprocess.preprocessor import get_df
from src.hpopt.hrl_optuna_utils import HyperparamsOptHRL

from src.config_training import TrainSettings

settings = TrainSettings()

TRAIN_START_DATE = settings.TRAIN_START_DATE
TRAIN_END_DATE = settings.TRAIN_END_DATE
VAL_START_DATE = settings.VAL_START_DATE
VAL_END_DATE = settings.VAL_END_DATE

INDICATORS = settings.INDICATORS

tickerlist = "DOW_30_red"

N_TRIALS = 40

df_train = get_df(
    start=TRAIN_START_DATE,
    end=TRAIN_END_DATE,
    tickerlist=tickerlist,
)
df_val = get_df(
    start=VAL_START_DATE,
    end=VAL_END_DATE,
    tickerlist=tickerlist,
)


hrl_opt = HyperparamsOptHRL(
    df_train=df_train, df_test=df_val, indicators=INDICATORS, n_trials=N_TRIALS
)
hrl_opt.run_opt()
