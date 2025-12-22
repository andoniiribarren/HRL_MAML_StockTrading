import pandas as pd
import numpy as np


def get_buyhold_df(
    df: pd.DataFrame,
    initial_cash: float = 1000000,
) -> pd.DataFrame:
    needed = {"date", "tic", "close_raw"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "tic"])

    prices = d.pivot(index="date", columns="tic", values="close_raw").sort_index()

    tickers = prices.columns.tolist()
    n = len(tickers)
    if n == 0:
        raise ValueError("No tickers found.")

    start_date = d["date"].min()
    alloc_per_asset = initial_cash / n
    start_prices = prices.loc[start_date]

    if start_prices.isna().any():
        raise ValueError(
            "Missing start prices. Use fill_method or choose a different start_date."
        )

    shares = np.floor(alloc_per_asset / start_prices)
    leftover_cash = float((alloc_per_asset - shares * start_prices).sum())

    portfolio_value = (prices.mul(shares, axis=1)).sum(axis=1) + leftover_cash

    portfolio_df = portfolio_value.reset_index()
    portfolio_df.columns = ["date", "account_value"]

    return portfolio_df
