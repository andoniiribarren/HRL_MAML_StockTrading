import importlib
import inspect

import numpy as np
import pandas as pd

from src.agent.base_rl_models import baseRLAgent
from src.agent.hrl_model import HRLAgent
from src.agent.meta.task_config import MetaTrainHelper
from src.env_stocktrading.trading_env_hrl import StockTradingEnvHRL
from src.preprocess.preprocessor import get_df


def _build_smoke_df(tickers: list[str], n_days: int, start_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    rows = []
    for day in range(n_days):
        date = start + pd.Timedelta(days=day)
        for idx, tic in enumerate(tickers):
            close_raw = 100.0 + 1.2 * idx + 0.35 * day
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


def _build_env(df: pd.DataFrame, indicators: list[str]) -> StockTradingEnvHRL:
    stock_dimension = len(df.tic.unique())
    state_space_manager = stock_dimension + len(indicators) * stock_dimension
    state_space_worker = 1 + 3 * stock_dimension

    return StockTradingEnvHRL(
        df=df,
        stock_dim=stock_dimension,
        hmax=10,
        initial_amount=100000,
        num_stock_shares=[0] * stock_dimension,
        buy_cost_pct=[0.001] * stock_dimension,
        sell_cost_pct=[0.001] * stock_dimension,
        state_space_M=state_space_manager,
        state_space_W=state_space_worker,
        action_space=stock_dimension,
        tech_indicator_list=indicators,
        make_plots=False,
        print_verbosity=100,
    )


def run_import_smoke() -> None:
    modules = [
        "src.agent.base_rl_models",
        "src.agent.hrl_model",
        "src.agent.meta_hrl_reptile",
        "src.agent.meta.task_config",
        "src.env_stocktrading.trading_env_hrl",
        "src.env_stocktrading.trading_env_rl",
        "src.hpopt.base_rl_hyperparam_op",
        "src.hpopt.hrl_optuna_utils",
        "src.preprocess.preprocessor",
        "src.eval.benchmarks",
        "src.eval.evaluate_functions",
    ]

    for module_name in modules:
        importlib.import_module(module_name)

    print("[OK] Import smoke")


def run_signature_smoke() -> None:
    get_df_sig = inspect.signature(get_df)
    expected_get_df = ["start", "end", "tickerlist", "json_path"]
    for name in expected_get_df:
        if name not in get_df_sig.parameters:
            raise AssertionError(f"Missing parameter '{name}' in get_df signature")

    train_model_sig = inspect.signature(baseRLAgent.train_model)
    expected_train_model = ["model", "tb_log_name", "total_timesteps", "callbacks"]
    for name in expected_train_model:
        if name not in train_model_sig.parameters:
            raise AssertionError(
                f"Missing parameter '{name}' in baseRLAgent.train_model signature"
            )

    train_hrl_sig = inspect.signature(HRLAgent.trainHRL)
    expected_train_hrl = [
        "total_timesteps",
        "reset_timesteps",
        "freeze_M",
        "freeze_W",
        "only_alignment_rew",
    ]
    for name in expected_train_hrl:
        if name not in train_hrl_sig.parameters:
            raise AssertionError(f"Missing parameter '{name}' in HRLAgent.trainHRL")

    task_helper_sig = inspect.signature(MetaTrainHelper.__init__)
    if "tickerlist" not in task_helper_sig.parameters:
        raise AssertionError("Missing parameter 'tickerlist' in MetaTrainHelper.__init__")

    print("[OK] Signature smoke")


def run_hrl_shape_smoke() -> None:
    indicators = ["macd", "rsi_30", "cci_30"]
    df = _build_smoke_df(["AAPL", "MSFT", "NVDA"], n_days=8, start_date="2024-01-01")
    env = _build_env(df, indicators)

    obs, _ = env.reset()
    stock_dim = len(df.tic.unique())
    expected_full_dim = 1 + 2 * stock_dim + len(indicators) * stock_dim
    expected_manager_dim = stock_dim + len(indicators) * stock_dim
    expected_worker_dim = 1 + 3 * stock_dim

    if len(obs["full_state"]) != expected_full_dim:
        raise AssertionError(
            f"full_state shape mismatch: expected {expected_full_dim}, got {len(obs['full_state'])}"
        )
    if len(obs["manager"]) != expected_manager_dim:
        raise AssertionError(
            f"manager shape mismatch: expected {expected_manager_dim}, got {len(obs['manager'])}"
        )
    if len(obs["worker"]) != expected_worker_dim:
        raise AssertionError(
            f"worker shape mismatch: expected {expected_worker_dim}, got {len(obs['worker'])}"
        )

    manager_kwargs = {
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "gamma": 0.99,
        "K_epochs": 4,
        "eps_clip": 0.2,
        "update_timestep": 2,
    }
    worker_kwargs = {
        "learning_rate": 1e-3,
        "tau": 0.005,
        "buffer_size": 500,
        "batch_size": 16,
        "gamma": 0.99,
        "verbose": 0,
    }

    agent = HRLAgent(
        env=env,
        stock_dim=stock_dim,
        manager_kwargs=manager_kwargs,
        worker_kwargs=worker_kwargs,
        initial_manager_timesteps=2,
        initial_worker_timesteps=2,
        n_alt_cycles=1,
        initial_cycle_steps=2,
    )

    agent.trainHRL(
        total_timesteps=2,
        reset_timesteps=True,
        freeze_M=True,
        freeze_W=True,
        only_alignment_rew=True,
        reset_ep=True,
    )

    account_memory, actions_memory, last_state = agent.predictHRL(env)

    if len(last_state) != expected_full_dim:
        raise AssertionError(
            f"Last state shape mismatch: expected {expected_full_dim}, got {len(last_state)}"
        )
    if account_memory is None or len(account_memory) == 0:
        raise AssertionError("Account memory is empty after predictHRL")
    if actions_memory is None or len(actions_memory) == 0:
        raise AssertionError("Actions memory is empty after predictHRL")

    print("[OK] HRL shape smoke")


def main() -> int:
    checks = [
        ("Import smoke", run_import_smoke),
        ("Signature smoke", run_signature_smoke),
        ("HRL shape smoke", run_hrl_shape_smoke),
    ]

    failures = []
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")

    if failures:
        print("\nSmoke checks failed:")
        for name, error in failures:
            print(f"- {name}: {error}")
        return 1

    print("\nAll Phase 2 smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
