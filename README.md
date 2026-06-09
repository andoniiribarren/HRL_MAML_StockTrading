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
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
├── configs/
│   ├── defaults.yaml
│   └── best_hyperparams.yaml
├── demo/
│   ├── eval.ipynb
│   ├── test_demo_hrl.ipynb
│   ├── test_demo_metahrl.ipynb
│   └── test_demo_rl_base.ipynb
├── tests/
│   └── smoke_phase2.py
└── src/
    ├── config_training.py
    ├── agent/
    │   ├── base_rl_models.py
    │   ├── hrl_model.py
    │   ├── meta_hrl_reptile.py
    │   ├── ppo.py
    │   └── meta/
    │       └── task_config.py
    ├── env_stocktrading/
    │   ├── trading_env_hrl.py
    │   └── trading_env_rl.py
    ├── eval/
    │   ├── benchmarks.py
    │   └── evaluate_functions.py
    ├── hpopt/
    │   ├── base_rl_hyperparam_op.py
    │   ├── hrl_hp_opt.py
    │   └── hrl_optuna_utils.py
    └── preprocess/
        ├── preprocessor.py
        └── tickers/
            └── ticker_lists.json
```

---

## Main components


### 1- Trading environments (src/env_stocktrading/)

trading_env_RL.py: baseline trading environment for a flat RL agent.

trading_env_HRL.py: HRL-compatible environment variant, designed to support manager/worker interaction.

Both environments are based on real historical market data and are designed in a gym-style interface. Adapted from FinRL environment.

### 2- Agents (src/agent/)

base_rl_models.py: baseline RL agent implementation (e.g., A2C via Stable-Baselines3).

hrl_model.py: hierarchical agent implementation (manager + worker training logic).

ppo.py: custom PPO implementation used for the manager (editable policy needed for meta-learning).

meta_hrl_reptile.py: MetaHRL agent implementation using a Reptile-style meta-learning loop on the manager.

meta/task_config.py: configuration and task definitions for meta-training regimes.

### 3- Preprocessing (src/preprocess/)

preprocessor.py: data downloading and preprocessing pipeline (Yahoo Finance via yfinance), feature engineering (technical indicators), and dataset preparation.

tickers/ticker_lists.json: ticker universe configuration (e.g., Dow Jones 30).

### 4- Hyperparameter optimization (src/hpopt/)

base_rl_hyperparam_op.py: Optuna search for baseline agent hyperparameters.

hrl_hp_opt.py + hrl_optuna_utils.py: Optuna utilities and search logic for HRL components.

### 5- Evaluation (src/eval/)

benchmarks.py: benchmark strategies (e.g., Buy & Hold, Minimum Variance).

evaluate_functions.py: evaluation utilities and metric computation (returns, Sharpe ratio, diversification, etc.).


## Installation

1) Create and activate a virtual environment

``` bash
python -m venv hrlenv
.\hrlenv\Scripts\Activate.ps1   # Windows
# source hrlenv/bin/activate     # Linux / macOS
```

2) Install dependencies
``` bash
pip install -r requirements.txt
```

Alternatively, install as a package (editable mode):
``` bash
pip install -e .
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

### Smoke tests

```bash
python -m tests.smoke_phase2
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
