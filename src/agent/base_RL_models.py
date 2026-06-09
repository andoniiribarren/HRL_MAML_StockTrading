from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from typing import Any, Callable

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}


class baseRLAgent:
    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        learning_rate=5e-5,
        gamma=0.99,
        max_grad_norm=0.5,
        n_steps=256,
        ent_coef=0.001,
        verbose=1,
        seed=None,
    ):
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in MODELS.")

        common_kwargs = {
            "policy": policy,
            "env": self.env,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "verbose": verbose,
            "seed": seed,
            "tensorboard_log": "./logs/",
        }

        # A2C/PPO expose rollout-specific parameters that off-policy algorithms do not.
        rollout_kwargs = {}
        if model_name in {"a2c", "ppo"}:
            rollout_kwargs = {
                "max_grad_norm": max_grad_norm,
                "n_steps": n_steps,
                "ent_coef": ent_coef,
            }

        return MODELS[model_name](
            **common_kwargs,
            **rollout_kwargs,
        )

    @staticmethod
    def _build_callback(callbacks):
        if callbacks is None:
            return None

        if isinstance(callbacks, BaseCallback):
            return callbacks

        if callable(callbacks):
            return callbacks

        callback_list = [callback for callback in callbacks if callback is not None]
        if not callback_list:
            return None
        if len(callback_list) == 1:
            return callback_list[0]

        return CallbackList(callback_list)

    @staticmethod
    def train_model(
        model: A2C | PPO | DDPG | TD3 | SAC,
        tb_log_name,
        total_timesteps=5000,
        callbacks: (
            BaseCallback
            | list[BaseCallback]
            | tuple[BaseCallback, ...]
            | Callable[[dict[str, Any], dict[str, Any]], bool]
            | None
        ) = None,
    ):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=baseRLAgent._build_callback(callbacks),
        )
        return model

    @staticmethod
    def predict_RL(model, environment, deterministic=True):
        """make a prediction and get results"""
        print("Starting prediction...")
        test_env, test_obs = environment.get_sb_env()
        account_memory = None
        actions_memory = None

        test_env.reset()
        max_steps = len(environment.df.dayorder.unique()) - 1

        for i in range(len(environment.df.dayorder.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            if i == max_steps - 1:
                last_state = test_obs
            test_obs, rewards, dones, info = test_env.step(action)

            if i == max_steps - 1:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0], last_state[0]
