import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime
import json


from preprocess.preprocessor import YahooDownloader
from env_stocktrading.trading_env_HRL import StockTradingEnvHRL
from agent.hrt import HRLforTrading
from preprocess.preprocessor import FeatureEngineer


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
    print_verbosity=1,
)


model = HRLforTrading(env=hrl_train_env, stock_dim=stock_dimension)
n_episodes = 10
trained_model = model.train_HRL_model()
