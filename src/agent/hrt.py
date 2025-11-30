from stable_baselines3 import DDPG
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FilterObservation
import numpy as np
import math
import torch as th

# obs_space = [balance, close prices_i, stock_shares_i, MACD_i, rsi30_i, cci30_i] --> 1 + stock_dim * 5
# obs_space_manager = [close prices_i, MACD_i, rsi30_i, cci30_i]
# obs_space_worker = [balance, close_prices_i, stock_shares_i, manager_actions_i]


class HRLforTrading():

    def __init__(
        self,
        env: gym.Env,
        manager_kwargs=None,
        worker_kwargs=None,
        initial_manager_timesteps: int = 2000,
        initial_worker_timesteps: int= 2000,
        n_alt_cycles: int = 500,
        initial_cycle_steps: int = 5000,
        dism_factor: float = 0.99
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
        self.env_M = FilterObservation(self.env, filter_keys=["manager"])
        self.env_W = FilterObservation(self.env, filter_keys=["worker"])
        # TODO ME FALTA ACTION WRAPPER

        self.manager = PPO(
                policy="MultiInputPolicy",
                env=self.env_M,
                **self.manager_kwargs,
            )
        
        self.worker = DDPG(
            policy="MultiInputPolicy",
            env=self.env_W,
            **self.worker_kwargs,
        )

    @staticmethod
    def alpha_function(t: int, alpha_0: float = 1.0, h: float = 0.001) -> float:
        return alpha_0 * math.exp(-h*t)

    def get_manager_reward(self, reward: float, r_align: np.ndarray, t: int | None, mode: str = "combined"):
        align_scalar = float(np.sum(r_align))
        if mode == "alignment_rew":
            return align_scalar
        elif mode == "combined":
            alpha_t = self.alpha_function(t)
            return alpha_t * align_scalar + (1 - alpha_t) * reward # Equation 2
        else:
            raise ValueError(f"'{mode}' is not a valid reward mode.")

    def trainHRL(self, total_timesteps: int, reset_timesteps: bool, freeze_M: bool, freeze_W: bool, only_alignment_rew: bool):
        num_timesteps = 0
        t_r=0

        # SETUP LEARNS
        # MANAGER
        _, callback = self.manager._setup_learn(
            total_timesteps,
            callback=None,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )

        # WORKER (DDPG)
        _, worker_callback = self.worker._setup_learn(
            total_timesteps,
            callback=None,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )

        obs, _ = self.env.reset()
        while num_timesteps < total_timesteps:
            self.manager.rollout_buffer.reset()

            for _ in range(self.manager.n_steps):
                # Separar observaciones
                obs_M = obs['manager']
                obs_W_raw = obs['worker']

                # Acciones del MANAGER
                with th.no_grad():
                    if isinstance(obs_M, np.ndarray):
                        if obs_M.ndim == 1:
                            obs_M_batch = obs_M[None, :] # Le metemos batch por error inicial
                        else:
                            obs_M_batch = obs_M
                    else:
                        # por si viniera como lista
                        obs_M_batch = np.array(obs_M, dtype=np.float32)[None, :]


                    obs_tensor = th.as_tensor(obs_M_batch).to(self.manager.device)
                    obs_M_dict = {"manager": obs_tensor}
                    actions_M_raw, values, log_probs = self.manager.policy(obs_M_dict) # antes obs

                # Generar obs para el WORKER
                actions_M_np = actions_M_raw.cpu().numpy()
                actions_M = actions_M_np - 1
                actions_M = np.squeeze(actions_M, axis=0) # Quitar dimensión extra

                obs_W = np.concatenate([obs_W_raw, actions_M], axis=-1)

                with th.no_grad():
                    actions_W, _ = self.worker.predict(obs_W, deterministic=False)

                actions_combined = actions_M * actions_W
                
                new_obs, reward, done, info = self.env.step(actions_combined)

                # Reward manager
                if isinstance(info, (list, tuple)):
                    info_0 = info[0]
                else:
                    info_0 = info

                rew_align = info_0.get("rew_align")
                if only_alignment_rew:
                    reward_M = self.get_manager_reward(reward, rew_align, t=None, mode="alignment_rew")
                else:
                    reward_M = self.get_manager_reward(reward, rew_align, t=t_r, mode="combined")
                    t_r += 1


                num_timesteps += self.env.num_envs
                self.manager.num_timesteps = num_timesteps
                self.worker.num_timesteps  = num_timesteps

                ###### Guardar en buffers ######
                # Manager
                self.manager.rollout_buffer.add(
                    obs_M,
                    actions_M_raw,
                    reward_M,
                    done,
                    values,
                    log_probs,
                )

                # Worker
                self.worker.replay_buffer.add(
                    obs_W,
                    new_obs['worker'],
                    actions_W,
                    reward,
                    done,
                    infos=None,
                )

                obs = new_obs

                # Actualizar política de worker si aplica
                if not freeze_W:
                    if (
                        self.worker.num_timesteps > self.worker.learning_starts
                        and self.worker.replay_buffer.size() > self.worker.batch_size
                    ):
                        # TODO revisar gradient step
                        self.worker.train(
                            gradient_steps=1,
                            batch_size=self.worker.batch_size,
                        )

            # Último valor
            with th.no_grad():
                obs_tensor = th.as_tensor(obs['manager']).to(self.manager.device)
                _, last_values, _ = self.manager.policy(obs_tensor)

            self.manager.rollout_buffer.compute_returns_and_advantage(last_values, done)

            # Actualizar políticas de manager si aplica
            if not freeze_M:
                self.manager.train()

        return self

    def ph1_manager_training(self):
        """
        Fase 1 - Entrenar manager solo con alignment rewards
        """
        print("Starting Phase 1: Only train Manager")
        # Aquí al comienzo usa sólo alignment
        self.trainHRL(total_timesteps=self.initial_manager_timesteps,
                      reset_timesteps=False,
                      freeze_M=False,
                      freeze_W=True,
                      only_alignment_rew=True)
        return self
        

    def ph2_worker_training(self):
        """
        Fase 2 - Worker training
        """
        print("Starting Phase 2: Only train Worker")
        # Congelar parámetros del manager y entrenar con ellos
        self.trainHRL(total_timesteps=self.initial_worker_timesteps,
                      reset_timesteps=False,
                      freeze_M=True,
                      freeze_W=False,
                      only_alignment_rew=True)
        return self
        

    def alternate_training(self):
        print("Starting Phase 3: Alternate training")
        steps_cycle = self.initial_cycle_steps

        for cycle in range(self.n_alt_cycles):
            
            self.trainHRL(total_timesteps=steps_cycle,
                      reset_timesteps=False,
                      freeze_M=False,
                      freeze_W=True,
                      only_alignment_rew=False)

            self.trainHRL(total_timesteps=steps_cycle,
                      reset_timesteps=False,
                      freeze_M=True,
                      freeze_W=False,
                      only_alignment_rew=True)

            steps_cycle = int(steps_cycle * self.dism_factor)

            print(f"[Cycle {cycle+1}/{self.n_alt_cycles}] done")
        
        return self


    def train_HRL_model(self):

        self.ph1_manager_training()
        self.ph2_worker_training()
        self.alternate_training()
        return self