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


from finHRL.agent.models import baseRLAgent
from finHRL.env_stocktrading.trading_env_RL import StockTradingEnv
from finHRL.preprocess.preprocessor import FeatureEngineer, YahooDownloader

class hyperparams_opt_RL:
    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 indicators: list[str],
                 n_episodes_train: int,
                 n_trials:int
                 ):
        self.indicators= indicators
        self.n_episodes_train = n_episodes_train
        self.n_trials = n_trials
        
        self.df_train = df_train
        self.df_test = df_test

        self.stock_dimension = len(self.df_train.tic.unique())
        # DF generation
        #self.df_train, self.df_test, self.stock_dimension = self.generate_data()


        self.episode_len = self.df_train.dayorder.nunique()

    """def generate_data(self):

        json_path = os.path.join(ROOT, "finHRL", "preprocess", "tickers", "ticker_lists.json")

        with open(json_path, "r") as f:
            data = json.load(f)

        if self.portfolio not in ["DOW_30", "CRYPTO_7"]:
            raise ValueError(f"The value '{self.portfolio}' is not permited. Portfolio possible values are 'DOW_30' or 'CRYPTO_7'.")

        tickers = data[self.portfolio]

        df = YahooDownloader(start_date = pd.to_datetime(self.train_start_date) - datetime.timedelta(days=30),
                        end_date = self.test_end_date,
                        ticker_list = tickers).fetch_data()

        fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = self.indicators,
                        use_turbulence=False,
                        user_defined_feature = False)

        processed = fe.preprocess_data(df)
        processed = processed.copy()
        processed = processed.fillna(0)
        processed = processed.replace(np.inf,0)

        processed = processed[processed.date >= self.train_start_date].reset_index(drop=True)

        stock_dimension = len(processed.tic.unique())

        df_train = processed[processed.date < self.test_start_date]
        df_test = processed[processed.date >= self.test_start_date]

        df_train["dayorder"] = df_train["date"].astype("category").cat.codes
        df_test["dayorder"] = df_test["date"].astype("category").cat.codes

        return df_train, df_test, stock_dimension"""

    def make_env(self, df: pd.DataFrame) -> StockTradingEnv:
        state_space = 1 + 2*self.stock_dimension + len(self.indicators)*self.stock_dimension
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
            print_verbosity=1
        )

    def objective(self, trial):
        gamma = trial.suggest_float('gamma', 0.98, 0.999, log=True)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 5, log=True)
        n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256])
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)
        ent_coef = trial.suggest_float('ent_coef', 1e-5, 0.01, log=True)

        train_env = self.make_env(self.df_train)
        test_env  = self.make_env(self.df_test)

        agent = baseRLAgent(env=train_env)

        model = agent.get_model("a2c",
                                learning_rate = learning_rate,
                                gamma = gamma,
                                max_grad_norm = max_grad_norm,
                                n_steps = n_steps,
                                ent_coef = ent_coef,
                                verbose=1)


        trained_model = agent.train_model(
            model,
            tb_log_name="a2c_test1",
            total_timesteps= self.n_episodes_train*self.episode_len
        )

        df_account_value, _ = baseRLAgent.predict_RL(
            model=trained_model, 
            environment = test_env)

        return df_account_value.account_value.iloc[-1]
    

    def run_opt(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print (f'Best hiperparams: {study.best_params}')

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
