import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_portfolio_evolution(
    df_list: list[pd.DataFrame],
    label_list: list[str],
    title: str = "Comparison",
    save: bool = False,
    filename: str = "plot_test",
):

    plt.figure(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (df, label) in enumerate(zip(df_list, label_list)):
        data_to_plot = df["account_value"]
        color = colors[i % len(colors)]
        plt.plot(data_to_plot, label=label, linewidth=2, color=color, alpha=0.9)

    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio value ($)", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    if save:
        file_name_ext = filename + ".png"
        plt.savefig(file_name_ext)


def calculate_portfolio_metrics(df_account, final_asset_state, risk_free_rate=0.0):
    vals = df_account["account_value"]

    initial_value = vals.iloc[0]
    final_balance = vals.iloc[-1]
    cumulative_return = (final_balance - initial_value) / initial_value

    daily_returns = vals.pct_change().dropna()

    if daily_returns.std() != 0:
        sharpe_ratio = (
            (daily_returns.mean() - risk_free_rate) / daily_returns.std()
        ) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    state_vector = np.array(final_asset_state)
    threshold = 1e-3
    active_assets = np.sum(np.abs(state_vector) > threshold)
    total_assets = len(state_vector)

    if total_assets > 0:
        diversification_score = (active_assets / total_assets) * 100
    else:
        diversification_score = 0.0

    metrics = {
        "Final Balance ($)": round(final_balance, 2),
        "Cumulative Return (%)": round(cumulative_return * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 4),
        "Diversification (%)": round(diversification_score, 2),
        "Active Assets": f"{active_assets}/{total_assets}",
    }

    return pd.DataFrame([metrics])
