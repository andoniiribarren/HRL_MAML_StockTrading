import warnings
import os

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from src.agent.meta.task_config import MetaTrainHelper
from src.agent.meta_hrl_reptile import MetaHRLAgent
from src.config_training import TrainSettings
from src.env_stocktrading.trading_env_hrl import StockTradingEnvHRL
from src.preprocess.preprocessor import get_df

settings = TrainSettings()

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
            close_raw = 110.0 + 1.5 * idx + 0.4 * day
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "dayorder": day,
                    "tic": tic,
                    "close_raw": close_raw,
                    "close": close_raw,
                    "macd": 0.01 * (day + idx),
                    "rsi_30": 50.0 + day,
                    "cci_30": 90.0 + idx,
                }
            )
    return pd.DataFrame(rows)


class _SmokeMetaTrainHelper:
    def __init__(self, base_df: pd.DataFrame, indicators: list[str]):
        self.base_df = base_df
        self.indicators = indicators
        self.task_ids = ["bearish", "bullish", "stagnant"]

    def sample_tasks(self, k: int = 3):
        return self.task_ids[:k]

    def _task_df(self, task: str) -> pd.DataFrame:
        scale = {"bearish": 0.98, "bullish": 1.02, "stagnant": 1.0}[task]
        df_task = self.base_df.copy()
        df_task["close_raw"] = df_task["close_raw"] * scale
        df_task["close"] = df_task["close"] * scale
        return df_task

    def create_env(self, task: str):
        df_train = self._task_df(task)
        stock_dimension = len(df_train.tic.unique())
        state_space_manager = stock_dimension + len(self.indicators) * stock_dimension
        state_space_worker = 1 + 3 * stock_dimension
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        return StockTradingEnvHRL(
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
            tech_indicator_list=self.indicators,
            make_plots=False,
            print_verbosity=100,
        )

if SMOKE_TEST:
    smoke_tickers = ["AAPL", "MSFT", "NVDA"]
    df_train = _build_smoke_df(smoke_tickers, n_days=8, start_date="2024-01-01")
    df_test = _build_smoke_df(smoke_tickers, n_days=6, start_date="2024-02-01")
else:
    df_train = get_df(start=TRAIN_START_DATE, end=TRAIN_END_DATE, tickerlist=tickerlist)
    df_test = get_df(start=TEST_START_DATE, end=TEST_END_DATE, tickerlist=tickerlist)

stock_dimension = len(df_train.tic.unique())


# TRAINING
episode_len = df_train.dayorder.nunique()
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

# Manager with PPO SB3
"""paper_params_manager = {
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "n_steps": 1024,  # no lo veo en PPO, buffer_size no existe
    "batch_size": 256,
    "gamma": 0.99,
    "verbose": 1,
}"""

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
    "verbose": 0,
}

initial_manager_episodes = 1 if SMOKE_TEST else 5
initial_worker_episodes = 1 if SMOKE_TEST else 2
initial_cycle_episodes = 1

task_helper = (
    _SmokeMetaTrainHelper(base_df=df_train, indicators=INDICATORS)
    if SMOKE_TEST
    else MetaTrainHelper(tickerlist=tickerlist)
)

model = MetaHRLAgent(
    task_helper=task_helper,
    env=hrl_train_env,
    stock_dim=stock_dimension,
    manager_kwargs=paper_params_manager,
    worker_kwargs=paper_params_worker,
    initial_manager_timesteps=initial_manager_episodes * episode_len,
    initial_worker_timesteps=initial_worker_episodes * episode_len,
    n_alt_cycles=1 if SMOKE_TEST else 2,
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
print(f"Max Drawdown:       {max_drawdown*100:.2f}%")
print(f"Volatilidad Anual:  {annual_volatility*100:.2f}%")
print("=" * 40)
