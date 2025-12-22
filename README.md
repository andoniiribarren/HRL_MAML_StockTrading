# HRL + Meta-Learning for Stock Trading

This repository contains the codebase developed for a Master’s thesis project focused on building and evaluating reinforcement learning agents for automated stock trading, including:
- a baseline deep RL agent,
- a hierarchical reinforcement learning (HRL) agent (manager/worker),
- and a meta-learning extension (Reptile-style) on top of the HRL agent (MetaHRL).

The project includes a custom gym-style trading environment built on real market data (Yahoo Finance) and an evaluation pipeline with classic financial benchmarks.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── demo
│   ├── eval.ipynb
│   ├── test_demo_HRL.ipynb
│   ├── test_demo_metaHRL.ipynb
│   └── test_demo_RLbase.ipynb
└── src
    ├── config_training.py
    ├── agent
    │   ├── base_RL_models.py
    │   ├── HRL_model.py
    │   ├── metaHRL_reptile.py
    │   ├── PPO.py
    │   └── meta
    │       └── task_config.py
    ├── env_stocktrading
    │   ├── trading_env_HRL.py
    │   └── trading_env_RL.py
    ├── eval
    │   ├── benchmarks.py
    │   └── evaluate_functions.py
    ├── hyperparameter_searching
    │   ├── base_RL_hyperparam_op.py
    │   ├── HRL_hp_opt.py
    │   └── HRL_optuna_utils.py
    └── preprocess
        ├── preprocessor.py
        └── tickers
            └── ticker_lists.json
```

---

## Main components


### 1- Trading environments (src/env_stocktrading/)

trading_env_RL.py: baseline trading environment for a flat RL agent.

trading_env_HRL.py: HRL-compatible environment variant, designed to support manager/worker interaction.

Both environments are based on real historical market data and are designed in a gym-style interface. Adapted from FinRL environment.

### 2- Agents (src/agent/)

base_RL_models.py: baseline RL agent implementation (e.g., A2C via Stable-Baselines3).

HRL_model.py: hierarchical agent implementation (manager + worker training logic).

PPO.py: custom PPO implementation used for the manager (editable policy needed for meta-learning).

metaHRL_reptile.py: MetaHRL agent implementation using a Reptile-style meta-learning loop on the manager.

meta/task_config.py: configuration and task definitions for meta-training regimes.

### 3- Preprocessing (src/preprocess/)

preprocessor.py: data downloading and preprocessing pipeline (Yahoo Finance via yfinance), feature engineering (technical indicators), and dataset preparation.

tickers/ticker_lists.json: ticker universe configuration (e.g., Dow Jones 30).

### 4- Hyperparameter optimization (src/hyperparameter_searching/)

base_RL_hyperparam_op.py: Optuna search for baseline agent hyperparameters.

HRL_hp_opt.py + HRL_optuna_utils.py: Optuna utilities and search logic for HRL components.

### 5- Evaluation (src/eval/)

benchmarks.py: benchmark strategies (e.g., Buy & Hold, Minimum Variance).

evaluate_functions.py: evaluation utilities and metric computation (returns, Sharpe ratio, diversification, etc.).


## Installation

1) Create and activate a virtual environment

``` bash
python -m venv hrlenv
.\hrlenv\Scripts\Activate.ps1
```

2) Install dependencies
``` bash
pip install -r requirements.txt
```

## Quick start

The easiest way to run experiments and reproduce results is through the notebooks in demo/.

### Baseline RL demo

- demo/test_demo_RLbase.ipynb

### HRL demo

- demo/test_demo_HRL.ipynb

### MetaHRL demo (Reptile)

- demo/test_demo_metaHRL.ipynb

### Evaluation demo

- demo/eval.ipynb


## License

This project is licensed under the MIT License.
