import warnings

import optuna
import pandas as pd

from agent.base_RL_models import baseRLAgent
from src.env_stocktrading.trading_env_RL import StockTradingEnv

warnings.filterwarnings("ignore")


class HyperparamsOptRL:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        indicators: list[str],
        n_episodes_train: int,
        n_trials: int,
    ):
        self.indicators = indicators
        self.n_episodes_train = n_episodes_train
        self.n_trials = n_trials

        self.df_train = df_train
        self.df_test = df_test

        self.stock_dimension = len(self.df_train.tic.unique())

        self.episode_len = self.df_train.dayorder.nunique()

    def make_env(self, df: pd.DataFrame) -> StockTradingEnv:
        state_space = (
            1 + 2 * self.stock_dimension + len(self.indicators) * self.stock_dimension
        )
        return StockTradingEnv(
            df=df,
            stock_dim=self.stock_dimension,
            hmax=100,
            initial_amount=1000000,
            num_stock_shares=[0] * self.stock_dimension,
            buy_cost_pct=[0.001] * self.stock_dimension,
            sell_cost_pct=[0.001] * self.stock_dimension,
            state_space=state_space,
            action_space=self.stock_dimension,
            tech_indicator_list=self.indicators,
            make_plots=False,
            print_verbosity=1,
        )

    def objective(self, trial):
        gamma = trial.suggest_float("gamma", 0.98, 0.999, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5, log=True)
        n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.01, log=True)

        train_env = self.make_env(self.df_train)
        test_env = self.make_env(self.df_test)

        agent = baseRLAgent(env=train_env)

        model = agent.get_model(
            "a2c",
            learning_rate=learning_rate,
            gamma=gamma,
            max_grad_norm=max_grad_norm,
            n_steps=n_steps,
            ent_coef=ent_coef,
            verbose=1,
        )

        trained_model = agent.train_model(
            model,
            tb_log_name="a2c_test1",
            total_timesteps=self.n_episodes_train * self.episode_len,
        )

        df_account_value, _, _ = baseRLAgent.predict_RL(
            model=trained_model, environment=test_env
        )

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
