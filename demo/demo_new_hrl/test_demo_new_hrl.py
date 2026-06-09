"""
Demo HRL: Train → Validate → Test pipeline.
Saves training params and final metrics to demo/results/.
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from src.agent.hrl_model import HRLAgent
from src.config_training import TrainSettings
from src.env_stocktrading.trading_env_hrl import StockTradingEnvHRL
from src.eval.evaluate_functions import calculate_portfolio_metrics
from src.preprocess.preprocessor import get_df

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
settings = TrainSettings()

TRAIN_START_DATE = settings.TRAIN_START_DATE
TRAIN_END_DATE = settings.TRAIN_END_DATE
VAL_START_DATE = settings.VAL_START_DATE
VAL_END_DATE = settings.VAL_END_DATE
TEST_START_DATE = settings.TEST23_START_DATE
TEST_END_DATE = settings.TEST23_END_DATE
INDICATORS = settings.INDICATORS

TICKERLIST = "DOW_30"
TEST24_START_DATE = settings.TEST24_START_DATE
TEST24_END_DATE = settings.TEST24_END_DATE

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("demo", "demo_new_hrl", "results", RUN_ID)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
hp_hrl = settings.best_hiperparams_HRL

manager_params = {
    "lr_actor": hp_hrl["lr_actor_M"],
    "lr_critic": hp_hrl["lr_critic_M"],
    "gamma": hp_hrl["gamma_M"],
    "K_epochs": 80,
    "eps_clip": 0.2,
    "update_timestep": hp_hrl["update_timestep"],
    "alpha_decay": 1.5e-5,
}

worker_params = {
    "learning_rate": hp_hrl["lr_W"],
    "gamma": hp_hrl["gamma_W"],
    "buffer_size": hp_hrl["buffer_size"],
    "batch_size": hp_hrl["batch_size"],
    "tau": 0.005,
    "learning_starts": 10000,
    "noise_sigma": 0.1,
}

# Training schedule (in episodes)
INITIAL_MANAGER_EPISODES = 30
INITIAL_WORKER_EPISODES = 40
N_ALT_CYCLES = 20
CYCLE_EPISODES = 10

# ── Data ─────────────────────────────────────────────────────────────────────
print("Downloading / loading data...")
df_train = get_df(start=TRAIN_START_DATE, end=TRAIN_END_DATE, tickerlist=TICKERLIST)
df_val = get_df(start=VAL_START_DATE, end=VAL_END_DATE, tickerlist=TICKERLIST)
df_test23 = get_df(start=TEST_START_DATE, end=TEST_END_DATE, tickerlist=TICKERLIST)
df_test24 = get_df(start=TEST24_START_DATE, end=TEST24_END_DATE, tickerlist=TICKERLIST)

# ── Env dimensions ───────────────────────────────────────────────────────────
stock_dimension = len(df_train.tic.unique())
state_space_manager = stock_dimension + len(INDICATORS) * stock_dimension
state_space_worker = 1 + 3 * stock_dimension
episode_len = df_train.dayorder.nunique()

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

ENV_KWARGS = dict(
    stock_dim=stock_dimension,
    hmax=100,
    initial_amount=1_000_000,
    num_stock_shares=num_stock_shares,
    buy_cost_pct=buy_cost_list,
    sell_cost_pct=sell_cost_list,
    state_space_M=state_space_manager,
    state_space_W=state_space_worker,
    action_space=stock_dimension,
    tech_indicator_list=INDICATORS,
    make_plots=False,
    print_verbosity=1,
)


def make_env(df):
    return StockTradingEnvHRL(df=df, **ENV_KWARGS)


# ── Training ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   HRL TRAINING")
print("=" * 60)
print(f"  Stocks: {stock_dimension} | Episode len: {episode_len} steps")
print(
    f"  Ph1 Manager: {INITIAL_MANAGER_EPISODES} eps ({INITIAL_MANAGER_EPISODES * episode_len} steps)"
)
print(
    f"  Ph2 Worker:  {INITIAL_WORKER_EPISODES} eps ({INITIAL_WORKER_EPISODES * episode_len} steps)"
)
print(f"  Ph3 Alternate: {N_ALT_CYCLES} cycles × {CYCLE_EPISODES} eps/sub-phase")
print("=" * 60)

train_env = make_env(df_train)

model = HRLAgent(
    env=train_env,
    stock_dim=stock_dimension,
    manager_kwargs=manager_params,
    worker_kwargs=worker_params,
    initial_manager_timesteps=INITIAL_MANAGER_EPISODES * episode_len,
    initial_worker_timesteps=INITIAL_WORKER_EPISODES * episode_len,
    n_alt_cycles=N_ALT_CYCLES,
    initial_cycle_steps=CYCLE_EPISODES * episode_len,
    tb_log=os.path.join("demo", "demo_new_hrl", "logs"),
)

LOG_NAME = f"hrl_new_demo_{RUN_ID}"

t_start = time.time()
trained_model = model.train_HRL_model(tb_log_name=LOG_NAME)
training_time = time.time() - t_start

print(f"\nTraining finished in {training_time / 60:.1f} min")


# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate_on(df, label):
    env = make_env(df)
    acc_mem, actions_mem, last_state = trained_model.predictHRL(env)

    if isinstance(acc_mem, list):
        df_account = pd.DataFrame(acc_mem, columns=["account_value"])
    else:
        df_account = acc_mem.copy()
        if "account_value" not in df_account.columns:
            df_account.columns = ["account_value"]

    metrics_df = calculate_portfolio_metrics(df_account, last_state)
    metrics = metrics_df.iloc[0].to_dict()

    # Extra metrics
    daily_returns = df_account["account_value"].pct_change().dropna()
    roll_max = df_account["account_value"].cummax()
    max_drawdown = (df_account["account_value"] / roll_max - 1.0).min()
    annual_vol = daily_returns.std() * np.sqrt(252)

    metrics["Max Drawdown (%)"] = round(max_drawdown * 100, 2)
    metrics["Annual Volatility (%)"] = round(annual_vol * 100, 2)

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    for k, v in metrics.items():
        print(f"  {k:.<30} {v}")

    # Save account df
    csv_path = os.path.join(
        RESULTS_DIR, f"cuenta_{label.lower().replace(' ', '_')}_HRL_new.csv"
    )
    df_account.to_csv(csv_path, index=False)

    return metrics


# ── Validation ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   EVALUATION")
print("=" * 60)

metrics_val = evaluate_on(df_val, "Validation (2022-2023)")
metrics_test23 = evaluate_on(df_test23, "Test (2023-2024)")
metrics_test24 = evaluate_on(df_test24, "Test (2024-2025)")

# ── Save run summary ─────────────────────────────────────────────────────────
run_summary = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_time_min": round(training_time / 60, 2),
    "total_timesteps": model.num_timesteps,
    "episodes": model._episode_count,
    "hyperparameters": {
        "manager": manager_params,
        "worker": {k: v for k, v in worker_params.items()},
        "schedule": {
            "initial_manager_episodes": INITIAL_MANAGER_EPISODES,
            "initial_worker_episodes": INITIAL_WORKER_EPISODES,
            "n_alt_cycles": N_ALT_CYCLES,
            "cycle_episodes": CYCLE_EPISODES,
            "episode_len": episode_len,
        },
    },
    "metrics": {
        "validation": metrics_val,
        "test_2023": metrics_test23,
        "test_2024": metrics_test24,
    },
}

summary_path = os.path.join(RESULTS_DIR, "run_summary_hrl_new.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'=' * 60}")
print(f"  Results saved to: {RESULTS_DIR}/")
print(f"    - run_summary_hrl_new.json")
print(f"    - cuenta_validation_*.csv / cuenta_test_*.csv")
print(f"  TensorBoard: tensorboard --logdir demo/demo_new_hrl/logs/{LOG_NAME}")
print(f"{'=' * 60}")
