import warnings

warnings.filterwarnings("ignore")

from HRL_utils import get_dfs

N_TRIALS = 40

df_train, df_test, stock_dimension, indicators = get_dfs()

from HRL_utils import hyperparams_opt_HRL

hrl_opt = hyperparams_opt_HRL(
    df_train=df_train, df_test=df_test, indicators=indicators, n_trials=N_TRIALS
)

hrl_opt.run_opt()
