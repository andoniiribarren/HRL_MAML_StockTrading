from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}


class baseRLAgent:
    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        learning_rate = 5e-5,
        gamma = 0.99,
        max_grad_norm = 0.5,
        n_steps = 256,
        ent_coef = 0.001,
        verbose=1,
        seed=None
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )
        
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            learning_rate = learning_rate,
            gamma = gamma,
            max_grad_norm = max_grad_norm,
            n_steps = n_steps,
            ent_coef = ent_coef,
            verbose=verbose,
            seed=seed, 
            tensorboard_log="./logs/"
        )
    
    @staticmethod
    def train_model(
        model: A2C | PPO | DDPG | TD3 | SAC,
        tb_log_name,
        total_timesteps=5000,
        callbacks: type[BaseCallback] = None,
    ):  # this function is static method, so it can be called without creating an instance of the class
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [callback for callback in callbacks]
                )
                if callbacks is not None
                else None
            ),
        )
        return model
    
    @staticmethod
    def predict_RL(model, environment, deterministic=True):
        """make a prediction and get results"""
        print("Starting prediction...")
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(environment.df.dayorder.unique()) - 1

        for i in range(len(environment.df.dayorder.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):  # more descriptive condition for early termination to clarify the logic
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]




################### DESARROLLO POSTERIOR ###################

class HRLAgent:

    def __init__(self, env):
        self.env = env


class HRLMAMLAgent:

    def __init__(self, env):
        self.env = env


