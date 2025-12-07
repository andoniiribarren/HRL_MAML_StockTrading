from stable_baselines3 import DDPG
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FilterObservation
import numpy as np
import math
import torch as th

from agent.HRL_model import HRLAgent

# obs_space = [balance, close prices_i, stock_shares_i, MACD_i, rsi30_i, cci30_i] --> 1 + stock_dim * 5
# obs_space_manager = [close prices_i, MACD_i, rsi30_i, cci30_i]
# obs_space_worker = [balance, close_prices_i, stock_shares_i, manager_actions_i]


class metaHRLAgent(HRLAgent):

    def __init__(
        self,
        env: gym.Env,
        stock_dim: int,
        manager_kwargs=None,
        worker_kwargs=None,
        initial_manager_timesteps: int = 2000,
        initial_worker_timesteps: int = 2000,
        n_alt_cycles: int = 500,
        initial_cycle_steps: int = 5000,
        dism_factor: float = 0.99,
    ):
        super().__init__(
            env,
            stock_dim,
            manager_kwargs,
            worker_kwargs,
            initial_manager_timesteps,
            initial_worker_timesteps,
            n_alt_cycles,
            initial_cycle_steps,
            dism_factor,
        )

    def metaTrainHRL(
        self,
        total_timesteps: int,
        reset_timesteps: bool,
        freeze_M: bool,
        freeze_W: bool,
        only_alignment_rew: bool,
        reset_env: bool = False,
    ):
        num_timesteps = 0
        t_r = 0

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

        episode_start = True
        obs, _ = self.env.reset()
        while num_timesteps < total_timesteps:
            self.manager.rollout_buffer.reset()

            for _ in range(self.manager.n_steps):
                if num_timesteps % 100 == 0:
                    print(f"Paso número: {num_timesteps}")
                # Separar observaciones
                obs_M = obs["manager"]
                obs_W_raw = obs["worker"]

                # Acciones del MANAGER
                with th.no_grad():
                    if isinstance(obs_M, np.ndarray):
                        if obs_M.ndim == 1:
                            obs_M_batch = obs_M[
                                None, :
                            ]  # Le metemos batch por error inicial
                        else:
                            obs_M_batch = obs_M
                    else:
                        # por si viniera como lista
                        obs_M_batch = np.array(obs_M, dtype=np.float32)[None, :]

                    obs_tensor = th.as_tensor(obs_M_batch).to(self.manager.device)
                    obs_M_dict = {"manager": obs_tensor}
                    actions_M_raw, values, log_probs = self.manager.policy(obs_M_dict)

                # Generar obs para el WORKER
                actions_M = actions_M_raw.cpu().numpy() - 1
                actions_M = np.squeeze(actions_M, axis=0)  # Quitar dimensión extra

                # WORKER: Actualizar anterior buffer y política del worker porque necesitábamos new_ibs
                if num_timesteps > 0:
                    new_obs_worker[-self.stock_dim :] = actions_M

                    new_obs_worker_dict = {"worker": new_obs_worker}

                    self.worker.replay_buffer.add(
                        prev_obs_dict_worker,
                        new_obs_worker_dict,
                        actions_W,
                        reward,
                        done,
                        infos=[{}],
                    )
                    # Actualizar política de worker si aplica
                    if not freeze_W:
                        if (
                            self.worker.num_timesteps > self.worker.learning_starts
                            and self.worker.replay_buffer.size()
                            > self.worker.batch_size
                        ):
                            # TODO revisar gradient step
                            self.worker.train(
                                gradient_steps=1,
                                batch_size=self.worker.batch_size,
                            )

                obs_W_raw = np.array(obs_W_raw, dtype=np.float32)

                # obs_W = np.concatenate([obs_W_raw, actions_M], axis=-1)
                obs_W_raw[-self.stock_dim :] = actions_M

                with th.no_grad():
                    obs_W_dict = {"worker": obs_W_raw}
                    actions_W, _ = self.worker.predict(obs_W_dict, deterministic=False)

                actions_combined = actions_M * actions_W

                new_obs, reward, done, _, info = self.env.step(actions_combined)

                # Reward manager
                if isinstance(info, (list, tuple)):
                    info_0 = info[0]
                else:
                    info_0 = info

                rew_align = info_0.get("rew_align")
                if only_alignment_rew:
                    reward_M = self.get_manager_reward(
                        reward, rew_align, t=None, mode="alignment_rew"
                    )
                else:
                    reward_M = self.get_manager_reward(
                        reward, rew_align, t=t_r, mode="combined"
                    )
                    t_r += 1

                num_timesteps += 1  # Revisar TODO self.env.num_envs
                self.manager.num_timesteps = num_timesteps
                self.worker.num_timesteps = num_timesteps

                ###### Guardar en buffers ######
                # Manager
                self.manager.rollout_buffer.add(
                    obs_M_dict,
                    actions_M_raw,
                    reward_M,
                    episode_start,
                    values,
                    log_probs,
                )

                prev_obs_dict_worker = obs_W_dict.copy()
                new_obs_worker = new_obs["worker"]
                obs = new_obs
                episode_start = done

                if done:
                    obs, _ = self.env.reset()

            # Último valor
            with th.no_grad():
                obs_M = obs["manager"]

                if isinstance(obs_M, np.ndarray):
                    obs_M_np = obs_M
                else:
                    obs_M_np = np.array(obs_M, dtype=np.float32)

                if obs_M_np.ndim == 1:
                    obs_M_batch = obs_M_np[None, :]
                else:
                    obs_M_batch = obs_M_np

                obs_tensor_M = th.as_tensor(obs_M_batch).to(self.manager.device)
                obs_M_dict = {"manager": obs_tensor_M}
                _, last_values, _ = self.manager.policy(obs_M_dict)

            last_dones = np.array([done], dtype=bool)
            self.manager.rollout_buffer.compute_returns_and_advantage(
                last_values, last_dones
            )

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
        self.trainHRL(
            total_timesteps=self.initial_manager_timesteps,
            reset_timesteps=False,
            freeze_M=False,
            freeze_W=True,
            only_alignment_rew=True,
        )
        return self

    def ph2_worker_training(self):
        """
        Fase 2 - Worker training
        """
        print("Starting Phase 2: Only train Worker")
        # Congelar parámetros del manager y entrenar con ellos
        self.trainHRL(
            total_timesteps=self.initial_worker_timesteps,
            reset_timesteps=False,
            freeze_M=True,
            freeze_W=False,
            only_alignment_rew=True,
        )
        return self

    def alternate_training(self):
        print("Starting Phase 3: Alternate training")
        steps_cycle = self.initial_cycle_steps

        for cycle in range(self.n_alt_cycles):

            self.trainHRL(
                total_timesteps=steps_cycle,
                reset_timesteps=False,
                freeze_M=False,
                freeze_W=True,
                only_alignment_rew=False,
            )

            self.trainHRL(
                total_timesteps=steps_cycle,
                reset_timesteps=False,
                freeze_M=True,
                freeze_W=False,
                only_alignment_rew=True,
            )

            steps_cycle = int(steps_cycle * self.dism_factor)

            print(f"[Cycle {cycle+1}/{self.n_alt_cycles}] done")

        return self

    def train_HRL_model(self):

        self.ph1_manager_training()
        self.ph2_worker_training()
        self.alternate_training()
        return self
