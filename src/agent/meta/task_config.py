import datetime
import json
from enum import Enum
import random

import numpy as np
import pandas as pd

from env_stocktrading.trading_env_HRL import StockTradingEnvHRL
from preprocess.preprocessor import get_df


class MarketTask(Enum):
    BEARISH = "Bearish"
    BULLISH = "Bullish"
    STAGNANT = "Stagnant"


class MetaTrainHelper:
    def __init__(self):

        # Mercado bajista
        self.task1 = {
            "title": "Bearish",
            "time_frame_start": "2022-01-01",
            "time_frame_end": "2023-01-01",
        }

        # Mercado alcista
        self.task2 = {
            "title": "Bullish",
            "time_frame_start": "2018-01-01",
            "time_frame_end": "2020-01-01",
        }

        # Mercado estancado
        self.task3 = {
            "title": "Stagnant",
            "time_frame_start": "2015-01-01",
            "time_frame_end": "2017-01-01",
        }

        df_t1 = get_df(
            start=self.task1["time_frame_start"], end=self.task1["time_frame_end"]
        )
        df_t2 = get_df(
            start=self.task2["time_frame_start"], end=self.task2["time_frame_end"]
        )
        df_t3 = get_df(
            start=self.task3["time_frame_start"], end=self.task3["time_frame_end"]
        )

        self.task1["df_train"] = df_t1
        self.task2["df_train"] = df_t2
        self.task3["df_train"] = df_t3

        self.stock_dimension = len(df_t1.tic.unique())

        self.tasks: dict[MarketTask, dict] = {
            MarketTask.BEARISH: self.task1,
            MarketTask.BULLISH: self.task2,
            MarketTask.STAGNANT: self.task3,
        }

    def sample_tasks(self, k: int = 3) -> list[MarketTask]:
        """Devuelve una lista de MarketTask, sampleadas sin reemplazo."""
        return random.sample(list(self.tasks.keys()), k)

    def create_env(self, task: MarketTask):

        task_cfg = self.tasks[task]
        df_train = task_cfg["df_train"]

        episode_len = df_train.dayorder.nunique()
        stock_dimension = len(df_train.tic.unique())
        INDICATORS = ["macd", "rsi_30", "cci_30"]

        state_space_manager = stock_dimension + len(INDICATORS) * stock_dimension
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
            tech_indicator_list=INDICATORS,
            make_plots=False,
            print_verbosity=1,
        )
