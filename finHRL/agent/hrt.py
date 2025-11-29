from stable_baselines3 import DDPG
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FilterObservation, TransformReward
import numpy as np
import math

# obs_space = [balance, close prices_i, stock_shares_i, MACD_i, rsi30_i, cci30_i] --> 1 + stock_dim * 5

# obs_space_manager = [close prices_i, MACD_i, rsi30_i, cci30_i]

# obs_space_worker = [balance, close_prices_i, stock_shares_i, manager_actions_i]   

class ManagerRewardWrapper(gym.Wrapper):
    def __init__(self, env, mode="alignment_rew"):
        super().__init__(env)
        self.mode = mode
        self.t = 0

    def set_mode(self, mode: str):
        """
        Possible modes:
          - alignment_rew: Only the alignment part. For the initial training.
          - combined: combined following Equation 2.
        """
        self.mode = mode

    @staticmethod
    def alpha_function(t, alpha_0: float = 1.0, h: float = 0.001) -> float:
        return alpha_0 * math.exp(-h*t)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r_align = info.get("rew_align", 0.0)
        r_worker   = reward

        if self.mode == "alignment_rew":
            reward_out = r_align
        elif self.mode == "combined":
            alpha_t = self.alpha_function(self.t)
            reward_out = alpha_t * r_align + (1 - alpha_t) * r_worker # Equation 2
        else:
            raise ValueError(f"'{self.mode}' is not a valid reward mode.")

        self.t += 1
        return obs, reward_out, terminated, truncated, info

def manager_reward_transform(reward):
    raise NotImplementedError

def worker_reward_transform(reward):
    raise NotImplementedError


#class Manager(PPO):
    


class HRLforTrading():

    def __init__(
        self,
        env,
        manager_kwargs=None,
        worker_kwargs=None,
        initial_manager_timesteps: int = 2000,
        initial_worker_timesteps: int= 2000,
        n_alt_cycles: int = 500,
        initial_cycle_steps: int = 5000,
        dism_factor: float = 0.98
    ):
        self.initial_manager_timesteps = initial_manager_timesteps
        self.initial_worker_timesteps = initial_worker_timesteps
        self.manager_kwargs = manager_kwargs or {}
        self.worker_kwargs  = worker_kwargs or {}
        self.n_alt_cycles = n_alt_cycles
        self.initial_cycle_steps = initial_cycle_steps
        self.dism_factor = dism_factor

        self.env = env

        # Apply obs transformations
        self.env_M = FilterObservation(self.env, filter_keys=["manager"]) # implementar en environment, ahora uno tipo Dict TODO
        self.env_W = FilterObservation(self.env, filter_keys=["worker"]) # Al worker se le añade la acción del manager, ojo

        # Apply reward transformations TODO en el environment tiene que devolver en info la reward align
        self.env_M = ManagerRewardWrapper(self.env, mode="alignment_rew")

        self.manager = PPO(
                policy="MlpPolicy",
                env=self.env_M,
                **manager_kwargs,
            )
        
        # TODO VER DE QUÉ MANER EL WORKER INCLUYE LA ACCIÓN DEL MANAGER
        self.worker = DDPG(
            policy="MlpPolicy",
            env=self.env_W,
            **worker_kwargs,
        )

    def ph1_manager_training(self):
        """
        Fase 1 - Entrenar manager solo con alignment rewards
        """
        # Aquí al comienzo usa sólo alignment
        self.manager.learn(
            total_timesteps=self.initial_manager_timesteps,
            reset_num_timesteps=False,
        )

    def ph2_worker_training(self):
        """
        Fase 2 - Worker training
        """
        # Congelar parámetros del manager y entrenar con ellos
        self.worker.learn(
            total_timesteps=self.initial_worker_timesteps,
            reset_num_timesteps=False,
        )

    def alternate_training(self):
        
        steps_cycle = self.initial_cycle_steps

        for cycle in range(self.n_alt_cycles):
            
            # TODO VER DE QUÉ FORMA AQUÍ ACTUAIZAMOS EL ALPHA PARA LA REWARD DEL MANAGER
            self.manager.learn(
                total_timesteps=steps_cycle,
                reset_num_timesteps=False,
            )

            self.worker.learn(
                total_timesteps=steps_cycle,
                reset_num_timesteps=False,
            )

            steps_cycle = int(steps_cycle * self.dism_factor)

            print(f"[Cycle {cycle+1}/{self.n_alt_cycles}] done")


    def HRL_train(self):

        print("Starting training:")
        print("Phase 1: Manager initial training")
        self.ph1_manager_training()

        print("Phase 2: Worker initial training")
        self.ph2_worker_training()

        print("Phase 3: Alternate training")
        self.alternate_training()



