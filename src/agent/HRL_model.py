from stable_baselines3 import DDPG
from agent.PPO import PPO
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FilterObservation
import numpy as np
import math
import torch as th

# obs_space = [balance, close prices_i, stock_shares_i, MACD_i, rsi30_i, cci30_i] --> 1 + stock_dim * 5
# obs_space_manager = [close prices_i, MACD_i, rsi30_i, cci30_i]
# obs_space_worker = [balance, close_prices_i, stock_shares_i, manager_actions_i]


class HRLAgent:

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
        self.initial_manager_timesteps = initial_manager_timesteps
        self.initial_worker_timesteps = initial_worker_timesteps
        self.manager_kwargs = manager_kwargs or {}
        self.worker_kwargs = worker_kwargs or {}
        self.n_alt_cycles = n_alt_cycles
        self.initial_cycle_steps = initial_cycle_steps
        self.dism_factor = dism_factor
        self.stock_dim = stock_dim

        self.env = env
        self.max_ep_len = len(self.env.df.dayorder.unique())

        # Apply obs transformations
        self.env_M = FilterObservation(self.env, filter_keys=["manager"])
        self.env_W = FilterObservation(self.env, filter_keys=["worker"])

        self.env_M.action_space = spaces.MultiDiscrete([3] * self.stock_dim)
        self.env_W.action_space = spaces.Box(low=0, high=1, shape=(stock_dim,))

        # args for PPO
        state_dim = self.env_M.observation_space["manager"].shape[0]
        action_dim = self.env_M.action_space.shape[0]
        # TODO mover a manager_kwargs fuera de aquí
        lr_actor = 3e-4
        lr_critic = 3e-4
        gamma = 0.99
        K_epochs = 80  # update policy for K epochs in one PPO update
        eps_clip = 0.2
        has_continuous_action_space = False
        action_std = 0.6

        self.manager = PPO(
            state_dim,
            action_dim,
            lr_actor,
            lr_critic,
            gamma,
            K_epochs,
            eps_clip,
            has_continuous_action_space,
            action_std,
        )

        self.worker = DDPG(
            policy="MultiInputPolicy",
            env=self.env_W,
            **self.worker_kwargs,
        )

        self.manager_timestep = 0
        self.worker.num_timesteps = 0
        self.num_timesteps = 0
        self.t_r = 0

    @staticmethod
    def alpha_function(t: int, alpha_0: float = 1.0, h: float = 0.001) -> float:
        return alpha_0 * math.exp(-h * t)

    def get_manager_reward(
        self, reward: float, r_align: np.ndarray, t: int | None, mode: str = "combined"
    ):
        if r_align is None:
            align_scalar = 0.0
        else:
            align_scalar = float(np.sum(r_align))
        if mode == "alignment_rew":
            return align_scalar
        elif mode == "combined":
            alpha_t = self.alpha_function(t)
            return alpha_t * align_scalar + (1 - alpha_t) * reward  # Equation 2
        else:
            raise ValueError(f"'{mode}' is not a valid reward mode.")

    def trainHRL(
        self,
        total_timesteps: int,
        reset_timesteps: bool,
        freeze_M: bool,
        freeze_W: bool,
        only_alignment_rew: bool,
        reset_ep: bool = False,
    ):
        num_train_timesteps = 0
        update_timestep = self.max_ep_len * 4

        # WORKER (DDPG)
        _, worker_callback = self.worker._setup_learn(
            total_timesteps,
            callback=None,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )

        episode_start = True
        if reset_ep:
            self.env.episode = 0

        while num_train_timesteps < total_timesteps:

            obs, _ = self.env.reset()
            current_ep_reward_manager = 0

            for t in range(1, self.max_ep_len + 1):

                if self.num_timesteps % 100 == 0:
                    print(f"Paso total número: {self.num_timesteps}")
                    print(f"Paso de este train número: {num_train_timesteps}")
                    print(f"Worker timesteps: {self.worker.num_timesteps}")
                    print(f"manager timesteps: {self.manager_timestep}\n")
                    print("DEBUG TR: ", self.t_r)
                # Separar observaciones
                obs_M = obs["manager"]
                obs_W_raw = obs["worker"]

                # Acciones del MANAGER
                actions_M_raw = self.manager.select_action(obs_M)

                ##### TODO
                # Generar obs para el WORKER
                actions_M = actions_M_raw.cpu().numpy() - 1
                actions_M = np.squeeze(actions_M, axis=0)  # Quitar dimensión extra
                ##### TODO

                # WORKER: Actualizar anterior buffer y política del worker porque necesitábamos new_ibs
                if num_train_timesteps > 0:
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
                        if self.num_timesteps % 100 == 0:
                            print("2- Worker timesteps:", self.worker.num_timesteps)
                            print(
                                "2- Worker buffer size:",
                                self.worker.replay_buffer.size(),
                            )
                        if (
                            self.worker.num_timesteps > self.worker.learning_starts
                            and self.worker.replay_buffer.size()
                            > self.worker.batch_size  # TODO: revisar en el repo de la asignatura DQN
                        ):
                            # TODO revisar gradient step
                            self.worker.train(
                                gradient_steps=1,
                                batch_size=self.worker.batch_size,
                            )

                obs_W_raw = np.array(obs_W_raw, dtype=np.float32)
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
                        reward, rew_align, t=self.t_r, mode="combined"
                    )
                    self.t_r += 1

                num_train_timesteps += 1
                self.num_timesteps += 1
                if not freeze_M:
                    self.manager_timestep += 1
                    if self.manager_timestep % update_timestep == 0:
                        self.manager.update()
                if not freeze_W:
                    self.worker.num_timesteps += 1

                ###### Guardar en buffers ######
                # Manager
                # saving reward and is_terminals
                self.manager.buffer.rewards.append(reward_M)
                self.manager.buffer.is_terminals.append(done)
                current_ep_reward += reward_M

                prev_obs_dict_worker = obs_W_dict.copy()
                new_obs_worker = new_obs["worker"]
                obs = new_obs
                episode_start = done

                if done:
                    break

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
            reset_ep=True,
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
                only_alignment_rew=False,  # A * Ralign + (1-A)*Rw igual un decaiminento más rápido
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
