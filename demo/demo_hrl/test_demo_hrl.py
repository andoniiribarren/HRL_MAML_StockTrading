import warnings
import os
import numpy as np
import pandas as pd

from src.agent.hrl_model import HRLAgent
from src.config_training import TrainSettings
from src.env_stocktrading.trading_env_hrl import StockTradingEnvHRL
from src.preprocess.preprocessor import get_df

settings = TrainSettings()
warnings.filterwarnings("ignore")

TRAIN_START_DATE = settings.TRAIN_START_DATE
TRAIN_END_DATE = settings.TRAIN_END_DATE
TEST_START_DATE = settings.TEST23_START_DATE
TEST_END_DATE = settings.TEST23_END_DATE
INDICATORS = settings.INDICATORS
tickerlist = "DOW_30_red"
SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"


def _build_smoke_df(tickers: list[str], n_days: int, start_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    rows = []
    for day in range(n_days):
        date = start + pd.Timedelta(days=day)
        for idx, tic in enumerate(tickers):
            close_raw = 100.0 + 2.0 * idx + 0.5 * day
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "dayorder": day,
                    "tic": tic,
                    "close_raw": close_raw,
                    "close": close_raw,
                    "macd": 0.01 * (day + idx),
                    "rsi_30": 45.0 + day,
                    "cci_30": 80.0 + idx,
                }
            )
    return pd.DataFrame(rows)

if SMOKE_TEST:
    smoke_tickers = ["AAPL", "MSFT", "NVDA"]
    df_train = _build_smoke_df(smoke_tickers, n_days=8, start_date="2024-01-01")
    df_test = _build_smoke_df(smoke_tickers, n_days=6, start_date="2024-02-01")
else:
    df_train = get_df(start=TRAIN_START_DATE, end=TRAIN_END_DATE, tickerlist=tickerlist)
    df_test = get_df(start=TEST_START_DATE, end=TEST_END_DATE, tickerlist=tickerlist)


# TRAINING
episode_len = df_train.dayorder.nunique()
stock_dimension = len(df_train.tic.unique())
state_space_manager = stock_dimension + len(INDICATORS) * stock_dimension
state_space_worker = 1 + 3 * stock_dimension

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

hrl_train_env = StockTradingEnvHRL(
    df=df_train,
    stock_dim=stock_dimension,
    hmax=100,
    initial_amount=1000000,
    num_stock_shares=num_stock_shares,
    buy_cost_pct=buy_cost_list,
    sell_cost_pct=sell_cost_list,
    state_space_M=state_space_manager,
    state_space_W=state_space_worker,
    action_space=stock_dimension,
    tech_indicator_list=INDICATORS,
    make_plots=False,
    print_verbosity=100 if SMOKE_TEST else 1,
)


paper_params_manager = {
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "gamma": 0.99,
    "K_epochs": 80,
    "eps_clip": 0.2,
    "update_timestep": 512,
}

paper_params_worker = {
    "learning_rate": 1e-3,
    "tau": 0.005,
    "buffer_size": 200000,
    "batch_size": 256,
    "gamma": 0.99,
    "verbose": 1,
}

initial_manager_episodes = 1 if SMOKE_TEST else 5
initial_worker_episodes = 1 if SMOKE_TEST else 2
initial_cycle_episodes = 1

model = HRLAgent(
    env=hrl_train_env,
    stock_dim=stock_dimension,
    manager_kwargs=paper_params_manager,
    worker_kwargs=paper_params_worker,
    initial_manager_timesteps=initial_manager_episodes * episode_len,
    initial_worker_timesteps=initial_worker_episodes * episode_len,
    # n_alt_cycles=500,
    n_alt_cycles=1,
    initial_cycle_steps=initial_cycle_episodes * episode_len,
)
trained_model = model.train_HRL_model()


hrl_test_env = StockTradingEnvHRL(
    df=df_test,
    stock_dim=stock_dimension,
    hmax=100,
    initial_amount=1000000,
    num_stock_shares=num_stock_shares,
    buy_cost_pct=buy_cost_list,
    sell_cost_pct=sell_cost_list,
    state_space_M=state_space_manager,
    state_space_W=state_space_worker,
    action_space=stock_dimension,
    tech_indicator_list=INDICATORS,
    make_plots=False,
    print_verbosity=100 if SMOKE_TEST else 1,
)

acc_mem, actions_mem, _ = trained_model.predictHRL(hrl_test_env)

if isinstance(acc_mem, list):
    df_account = pd.DataFrame(acc_mem, columns=["account_value"])
else:
    df_account = acc_mem.copy()
    if "account_value" not in df_account.columns:
        df_account.columns = ["account_value"]

df_account["daily_return"] = df_account["account_value"].pct_change()

initial_value = df_account["account_value"].iloc[0]
final_value = df_account["account_value"].iloc[-1]
profit = final_value - initial_value
cumulative_return = (final_value / initial_value) - 1

annual_factor = 252
daily_std = df_account["daily_return"].std()
if daily_std and not np.isnan(daily_std):
    sharpe_ratio = (df_account["daily_return"].mean() / daily_std) * np.sqrt(
        annual_factor
    )
else:
    sharpe_ratio = 0.0

roll_max = df_account["account_value"].cummax()
daily_drawdown = df_account["account_value"] / roll_max - 1.0
max_drawdown = daily_drawdown.min()

annual_volatility = df_account["daily_return"].std() * np.sqrt(annual_factor)

print("\n" + "=" * 40)
print(f"   RESULTADOS DE LA EVALUACIÓN (TEST)")
print("=" * 40)
print(f"Valor Inicial:      ${initial_value:,.2f}")
print(f"Valor Final:        ${final_value:,.2f}")
print(f"Beneficio Total:    ${profit:,.2f}")
print("-" * 40)
print(f"Retorno Acumulado:  {cumulative_return*100:.2f}%")
print(f"Sharpe Ratio:       {sharpe_ratio:.4f}")
print("=" * 40)
