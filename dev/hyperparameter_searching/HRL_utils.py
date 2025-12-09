import warnings

warnings.filterwarnings("ignore")

import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import optuna
import json
import pandas as pd
import numpy as np
import datetime


from preprocess.preprocessor import YahooDownloader
from preprocess.preprocessor import FeatureEngineer
from src.env_stocktrading.trading_env_HRL import StockTradingEnvHRL
from src.agent.HRL_model import HRLAgent


def get_dfs():
    with open("src/preprocess/tickers/ticker_lists.json", "r") as f:
        data = json.load(f)

    dow_30 = data["DOW_30"]
    cryptos = data["CRYPTO_7"]

    TRAIN_START_DATE = "2017-01-01"
    TRAIN_END_DATE = "2022-01-01"
    TEST_START_DATE = "2022-01-01"
    TEST_END_DATE = "2023-01-01"

    df = YahooDownloader(
        start_date=pd.to_datetime(TRAIN_START_DATE) - datetime.timedelta(days=30),
        end_date=TEST_END_DATE,
        ticker_list=dow_30,
    ).fetch_data()

    INDICATORS = ["macd", "rsi_30", "cci_30"]

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf, 0)

    processed = processed[processed.date >= TRAIN_START_DATE].reset_index(drop=True)

    stock_dimension = len(processed.tic.unique())

    df_train = processed[processed.date < TEST_START_DATE]
    df_test = processed[processed.date >= TEST_START_DATE]

    df_train["dayorder"] = df_train["date"].astype("category").cat.codes
    df_test["dayorder"] = df_test["date"].astype("category").cat.codes

    return df_train, df_test, stock_dimension, INDICATORS


class hyperparams_opt_HRL:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        indicators: list[str],
        n_trials: int,
    ):
        self.indicators = indicators
        self.n_trials = n_trials

        self.df_train = df_train
        self.df_test = df_test

        self.stock_dimension = len(self.df_train.tic.unique())
        # DF generation
        # self.df_train, self.df_test, self.stock_dimension = self.generate_data()

        self.episode_len = self.df_train.dayorder.nunique()

    def make_env(self, df: pd.DataFrame) -> StockTradingEnvHRL:

        state_space_manager = (
            self.stock_dimension + len(self.indicators) * self.stock_dimension
        )
        state_space_worker = 1 + 3 * self.stock_dimension

        buy_cost_list = sell_cost_list = [0.001] * self.stock_dimension
        num_stock_shares = [0] * self.stock_dimension

        return StockTradingEnvHRL(
            df=df,
            stock_dim=self.stock_dimension,
            hmax=100,
            initial_amount=1000000,
            num_stock_shares=num_stock_shares,
            buy_cost_pct=buy_cost_list,
            sell_cost_pct=sell_cost_list,
            state_space_M=state_space_manager,
            state_space_W=state_space_worker,
            action_space=self.stock_dimension,
            tech_indicator_list=self.indicators,
            make_plots=False,
            print_verbosity=1,
        )

    def objective(self, trial: optuna.Trial):
        """gamma = trial.suggest_float("gamma", 0.98, 0.999, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5, log=True)
        n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.01, log=True)"""

        lr_actor_M = trial.suggest_float("lr_actor_M", 5e-5, 1e-3, log=True)
        lr_critic_M = trial.suggest_float("lr_critic_M", 5e-5, 1e-3, log=True)
        gamma_M = trial.suggest_float("gamma_M", 0.98, 0.999, log=True)
        update_timestep = trial.suggest_categorical(
            "update_timestep", [256, 512, 1024, 2048, 4096]
        )

        gamma_W = trial.suggest_float("gamma_W", 0.98, 0.999, log=True)
        lr_W = trial.suggest_float("lr_W", 5e-5, 5e-3, log=True)
        buffer_size = trial.suggest_categorical(
            "buffer_size", [2000, 10000, 40000, 100000, 200000]
        )
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

        params_manager = {
            "lr_actor": lr_actor_M,
            "lr_critic": lr_critic_M,
            "gamma": gamma_M,
            "K_epochs": 80,
            "eps_clip": 0.2,
            "update_timestep": update_timestep,
        }

        params_worker = {
            "learning_rate": lr_W,
            "tau": 0.005,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma_W,
            "verbose": 1,
        }

        train_env = self.make_env(self.df_train)
        test_env = self.make_env(self.df_test)

        initial_manager_episodes = 12
        initial_worker_episodes = 10
        initial_cycle_episodes = 4

        model = HRLAgent(
            env=train_env,
            stock_dim=self.stock_dimension,
            manager_kwargs=params_manager,
            worker_kwargs=params_worker,
            initial_manager_timesteps=initial_manager_episodes * self.episode_len,
            initial_worker_timesteps=initial_worker_episodes * self.episode_len,
            n_alt_cycles=4,
            initial_cycle_steps=initial_cycle_episodes * self.episode_len,
            dism_factor=0.9,
        )

        trained_model = model.train_HRL_model()

        df_account_value, _ = trained_model.predictHRL(env=test_env)

        return df_account_value.account_value.iloc[-1]

    def run_opt(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print(f"Best hiperparams: {study.best_params}")

        t = study.best_trial
        print("========================")
        print("BEST TRIAL INFO")
        print("Best trial number:", t.number)
        print("Best value:", t.value)
        print("Best params:", t.params)
        print("Duration:", t.duration)

        df_trials = study.trials_dataframe()
        print("\n========================")
        print("ALL TRIALS INFO")
        print(df_trials)
