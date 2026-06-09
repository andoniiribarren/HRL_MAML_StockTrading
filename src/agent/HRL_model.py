from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from src.agent.ppo import PPO
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FilterObservation
import numpy as np
import os
import math
import torch as th


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
        tb_log: str | None = None,
    ):
        self.initial_manager_timesteps = initial_manager_timesteps
        self.initial_worker_timesteps = initial_worker_timesteps
        self.manager_kwargs = manager_kwargs or {}
        self.worker_kwargs = worker_kwargs or {}
        self.n_alt_cycles = n_alt_cycles
        self.initial_cycle_steps = initial_cycle_steps
        self.stock_dim = stock_dim

        self.env = env
        self.max_ep_len = len(self.env.df.dayorder.unique())

        # Apply obs transformations
        self.env_M = FilterObservation(self.env, filter_keys=["manager"])
        self.env_W = FilterObservation(self.env, filter_keys=["worker"])

        self.manager_obs_size = self.env_M.observation_space["manager"].shape[0]
        self.worker_obs_size = self.env_W.observation_space["worker"].shape[0]

        self.env_M.action_space = spaces.MultiDiscrete([3] * self.stock_dim)
        self.env_W.action_space = spaces.Box(low=0, high=1, shape=(stock_dim,))

        # args for PPO
        state_dim = self.env_M.observation_space["manager"].shape[0]
        action_dim = [3] * self.stock_dim

        self.alpha_decay = self.manager_kwargs.pop("alpha_decay", 1.5e-5)

        self.manager = PPO(
            state_dim,
            action_dim,
            lr_actor=self.manager_kwargs["lr_actor"],
            lr_critic=self.manager_kwargs["lr_critic"],
            gamma=self.manager_kwargs["gamma"],
            K_epochs=self.manager_kwargs["K_epochs"],
            eps_clip=self.manager_kwargs["eps_clip"],
        )

        # DDPG action_noise
        n = self.env.action_space.shape[0]
        self.worker_noise_sigma = self.worker_kwargs.pop("noise_sigma", 0.1)
        action_noise = NormalActionNoise(
            mean=np.zeros(n), sigma=self.worker_noise_sigma * np.ones(n)
        )
        self.worker_action_noise = action_noise

        worker_learning_starts = self.worker_kwargs.pop("learning_starts", 10000)
        self.worker = DDPG(
            policy="MultiInputPolicy",
            env=self.env_W,
            learning_starts=worker_learning_starts,
            action_noise=action_noise,
            **self.worker_kwargs,
        )

        self.manager_timestep = 0
        self.worker.num_timesteps = 0
        self.num_timesteps = 0
        self.t_r = 0

        self.tb_log = tb_log
        self._writer = None
        self._episode_count = 0

    @staticmethod
    def _ensure_vector(
        value,
        expected_size: int,
        label: str,
        dtype=None,
    ) -> np.ndarray:
        vector = np.asarray(value, dtype=dtype).reshape(-1)
        if vector.size != expected_size:
            raise ValueError(
                f"Invalid {label} shape: expected ({expected_size},), got {vector.shape}."
            )
        return vector

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
            alpha_t = self.alpha_function(t, h=self.alpha_decay)
            return alpha_t * align_scalar + (1 - alpha_t) * reward
        else:
            raise ValueError(f"'{mode}' is not a valid reward mode.")

    # ── TensorBoard logging helpers ──────────────────────────────────

    def _setup_writer(self, tb_log_name=None):
        if self.tb_log is not None:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(self.tb_log, tb_log_name or "HRL_1")
            self._writer = SummaryWriter(log_dir)

    def _log_scalar(self, tag, value, step):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def _close_writer(self):
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

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
        manager_update_timestep = self.manager_kwargs["update_timestep"]

        # WORKER (DDPG)
        _, worker_callback = self.worker._setup_learn(
            total_timesteps,
            callback=None,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )

        if reset_ep:
            self.env.episode = 0

        while num_train_timesteps < total_timesteps:

            obs, _ = self.env.reset()
            current_ep_reward_manager = 0
            current_ep_reward_env = 0
            current_ep_alignment = []

            for t in range(1, self.max_ep_len + 1):

                # if self.num_timesteps % 100 == 0:
                # print(f"Paso total número: {self.num_timesteps}")
                # print(f"Paso de este train número: {num_train_timesteps}")
                # print(f"Worker timesteps: {self.worker.num_timesteps}")
                # print(f"manager timesteps: {self.manager_timestep}\n")

                # Split obs
                obs_M = self._ensure_vector(
                    obs["manager"], self.manager_obs_size, "manager observation"
                )
                obs_W_raw = self._ensure_vector(
                    obs["worker"], self.worker_obs_size, "worker observation"
                )

                # MANAGER actions
                actions_M_raw = self.manager.select_action(obs_M, deterministic=freeze_M)

                # Create worker obs
                actions_M = (
                    self._ensure_vector(
                        actions_M_raw,
                        self.stock_dim,
                        "manager action",
                    ).astype(np.int32)
                    - 1
                )

                # Worker: update previous steps buffer because manager actions needed
                if num_train_timesteps > 0:
                    new_obs_worker[-self.stock_dim :] = actions_M

                    new_obs_worker_dict = {"worker": new_obs_worker}
                    if not freeze_W:
                        self.worker.replay_buffer.add(
                            prev_obs_dict_worker,
                            new_obs_worker_dict,
                            actions_W,
                            reward,
                            done,
                            infos=[{}],
                        )
                        if (
                            self.worker.num_timesteps > self.worker.learning_starts
                            and self.worker.replay_buffer.size()
                            > self.worker.batch_size
                        ):
                            self.worker.train(
                                gradient_steps=1,
                                batch_size=self.worker.batch_size,
                            )

                obs_W_raw = self._ensure_vector(
                    obs_W_raw,
                    self.worker_obs_size,
                    "worker observation",
                    dtype=np.float32,
                )
                obs_W_raw[-self.stock_dim :] = actions_M

                with th.no_grad():
                    obs_W_dict = {"worker": obs_W_raw}
                    actions_W, _ = self.worker.predict(
                        obs_W_dict, deterministic=freeze_W
                    )

                # Add exploration noise manually (predict() does not apply it)
                if not freeze_W:
                    noise = self.worker_action_noise()
                    actions_W = np.clip(actions_W + noise, 0.0, 1.0)

                actions_W = self._ensure_vector(
                    actions_W,
                    self.stock_dim,
                    "worker action",
                    dtype=np.float32,
                )

                actions_combined = actions_M * actions_W
                actions_combined = self._ensure_vector(
                    actions_combined,
                    self.stock_dim,
                    "combined action",
                    dtype=np.float32,
                )

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

                current_ep_reward_env += reward
                if rew_align is not None:
                    current_ep_alignment.append(float(np.mean(rew_align)))

                num_train_timesteps += 1
                self.num_timesteps += 1

                # Manager, saving reward and is_terminals, train if applies
                if not freeze_M:
                    self.manager.buffer.rewards.append(reward_M)
                    self.manager.buffer.is_terminals.append(done)
                    current_ep_reward_manager += reward_M
                    self.manager_timestep += 1
                    if self.manager_timestep % manager_update_timestep == 0:
                        ppo_metrics = self.manager.update()
                        self._log_scalar("ppo/loss", ppo_metrics["loss"], self.num_timesteps)
                        self._log_scalar("ppo/entropy", ppo_metrics["entropy"], self.num_timesteps)
                        self._log_scalar("ppo/ratio", ppo_metrics["ratio"], self.num_timesteps)
                if not freeze_W:
                    self.worker.num_timesteps += 1

                prev_obs_dict_worker = obs_W_dict.copy()
                new_obs_worker = self._ensure_vector(
                    new_obs["worker"],
                    self.worker_obs_size,
                    "next worker observation",
                    dtype=np.float32,
                ).copy()
                obs = new_obs

                if done:
                    break

            # ── Episode-end logging ──────────────────────────────────
            self._episode_count += 1
            fs = self.env.state["full_state"]
            portfolio_value = fs[0] + sum(
                np.array(fs[1 : self.stock_dim + 1])
                * np.array(fs[self.stock_dim + 1 : self.stock_dim * 2 + 1])
            )
            initial_value = self.env.asset_memory[0] if self.env.asset_memory else 1.0
            portfolio_return = (portfolio_value - initial_value) / initial_value * 100

            self._log_scalar("reward/manager_episode", current_ep_reward_manager, self.num_timesteps)
            self._log_scalar("reward/worker_episode", current_ep_reward_env, self.num_timesteps)
            if current_ep_alignment:
                self._log_scalar("reward/alignment_mean", np.mean(current_ep_alignment), self.num_timesteps)
            self._log_scalar("portfolio/value", portfolio_value, self.num_timesteps)
            self._log_scalar("portfolio/return_pct", portfolio_return, self.num_timesteps)
            self._log_scalar("alpha/alpha_t", self.alpha_function(self.t_r, h=self.alpha_decay), self.num_timesteps)
            if hasattr(self.worker, "replay_buffer") and self.worker.replay_buffer is not None:
                self._log_scalar("worker/replay_buffer_size", self.worker.replay_buffer.size(), self.num_timesteps)
            self._log_scalar("training/timesteps", self.num_timesteps, self.num_timesteps)
            self._log_scalar("training/episodes", self._episode_count, self.num_timesteps)

        return self

    def ph1_manager_training(self):
        print("Starting Phase 1: Only train Manager")
        self.manager.policy.train()
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
        print("Starting Phase 2: Only train Worker")
        self.manager.policy.eval()
        self.worker.replay_buffer.reset()
        self.worker_action_noise.reset()
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
            self.manager.policy.train()
            self.trainHRL(
                total_timesteps=steps_cycle,
                reset_timesteps=False,
                freeze_M=False,
                freeze_W=True,
                only_alignment_rew=False,
            )
            self.manager.policy.eval()
            self.trainHRL(
                total_timesteps=steps_cycle,
                reset_timesteps=False,
                freeze_M=True,
                freeze_W=False,
                only_alignment_rew=True,
            )

            print(f"[Cycle {cycle+1}/{self.n_alt_cycles}] done")

        return self

    def train_HRL_model(self, tb_log_name=None):
        self._setup_writer(tb_log_name)
        self.ph1_manager_training()
        self.ph2_worker_training()
        self.alternate_training()
        self._close_writer()
        return self

    def predictHRL(self, env):
        obs, _ = env.reset()
        done = False
        last_state = obs

        max_steps = len(env.df.dayorder.unique()) - 1

        account_memory = []
        actions_memory = []

        self.manager.policy.eval()

        for i in range(max_steps + 1):
            obs_M = self._ensure_vector(
                obs["manager"], self.manager_obs_size, "manager observation"
            )
            action_M_raw = self.manager.select_action(obs_M, deterministic=True)
            action_M = (
                self._ensure_vector(action_M_raw, self.stock_dim, "manager action").astype(
                    np.int32
                )
                - 1
            )

            obs_W_raw = self._ensure_vector(
                obs["worker"],
                self.worker_obs_size,
                "worker observation",
                dtype=np.float32,
            )

            new_obs_worker = np.array(obs_W_raw, dtype=np.float32)
            new_obs_worker[-self.stock_dim :] = action_M

            action_W, _ = self.worker.predict(
                {"worker": new_obs_worker}, deterministic=True
            )

            action_W = self._ensure_vector(
                action_W,
                self.stock_dim,
                "worker action",
                dtype=np.float32,
            )

            combined_action = action_M * action_W
            combined_action = self._ensure_vector(
                combined_action,
                self.stock_dim,
                "combined action",
                dtype=np.float32,
            )
            next_obs, reward, done, _, info = env.step(combined_action)

            # Save results
            if i == max_steps - 1 or done:
                print("Retrieving memory from env...")
                account_memory = env.save_asset_memory()
                actions_memory = env.save_action_memory()
                last_state = obs

                if done:
                    print("Hit end!")
                    break

            obs = next_obs
        return account_memory, actions_memory, last_state["full_state"]
