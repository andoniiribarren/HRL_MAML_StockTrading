import copy

import numpy as np
import torch
from stable_baselines3 import PPO
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from agent.meta.task_config import MetaTrainHelper
from agent.HRL_model import HRLAgent


class MetaHRLAgent(HRLAgent):
    def __init__(self, task_helper: MetaTrainHelper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_helper = task_helper

    def meta_train_manager(
        self,
        meta_epochs=50,
        meta_batch_size=3,
        meta_lr=0.001,
        inner_epochs=1,
        reset_ep=False,
    ):

        print(f"--- [Meta] Iniciando Reptile (Meta-LR: {meta_lr}) ---")
        self.manager.policy.train()
        ppo_update_step = self.manager_kwargs.get("update_timestep")

        for iteration in range(meta_epochs):

            # Guardar theta
            current_meta_state = self.manager.policy.state_dict()

            # Inicializar acumulador de pesos
            sum_adapted_weights = {
                k: torch.zeros_like(v) for k, v in current_meta_state.items()
            }

            tasks = self.task_helper.sample_tasks(k=meta_batch_size)

            for task in tasks:
                print(f"Task {task}, Meta Epoch: {iteration}")
                task_env = self.task_helper.create_env(task)

                # agent_clone = deepcopy(self)
                agent_clone = copy.copy(self)
                agent_clone.manager = copy.deepcopy(self.manager)

                agent_clone.env = task_env
                agent_clone.manager.buffer.clear()

                steps_needed = ppo_update_step * inner_epochs

                agent_clone.trainHRL(
                    total_timesteps=steps_needed,
                    reset_timesteps=True,
                    freeze_M=False,
                    freeze_W=True,
                    only_alignment_rew=False,
                    reset_ep=reset_ep,
                )

                clone_state = agent_clone.manager.policy.state_dict()
                for key in sum_adapted_weights:
                    sum_adapted_weights[key] += clone_state[key]

                del agent_clone

            avg_weights = {k: v / len(tasks) for k, v in sum_adapted_weights.items()}

            # Actualización Reptile: New = Old + Beta * (Avg - Old)
            new_meta_state = {}

            for key in current_meta_state:
                direction = avg_weights[key] - current_meta_state[key]
                new_meta_state[key] = current_meta_state[key] + meta_lr * direction

            # Poner en los pesos del agente
            self.manager.policy.load_state_dict(new_meta_state)
            self.manager.policy_old.load_state_dict(new_meta_state)

            if (iteration + 1) % 5 == 0:
                print(f"  > Meta-Epoch {iteration+1}/{meta_epochs} completada.")

    def ph1_manager_training(self):
        print("\n=== Phase 1: Meta-Training Manager ===")
        # En lugar de trainHRL, llamamos a Reptile
        # Convertimos tus timesteps iniciales a épocas aproximadas
        steps_per_epoch = (
            self.manager_kwargs.get("update_timestep", 2048) * 3
        )  # Aprox por batch
        n_epochs = max(5, int(self.initial_manager_timesteps / steps_per_epoch))

        self.meta_train_manager(
            meta_epochs=n_epochs, meta_batch_size=3, meta_lr=0.001, reset_ep=True
        )
        return self

    def alternate_training(self):
        print("\n=== Phase 3: Alternate Meta-Training ===")
        steps_cycle = self.initial_cycle_steps

        for cycle in range(self.n_alt_cycles):
            print(f"\n--- Cycle {cycle+1}/{self.n_alt_cycles} ---")

            ppo_steps = self.manager_kwargs.get("update_timestep", 2048)
            epochs_cycle = max(1, int(steps_cycle / ppo_steps))

            self.meta_train_manager(
                meta_epochs=epochs_cycle, meta_batch_size=3, meta_lr=0.001
            )

            # 2. Turno Worker (STANDARD)
            self.trainHRL(
                total_timesteps=steps_cycle,
                reset_timesteps=False,
                freeze_M=True,
                freeze_W=False,
                only_alignment_rew=True,
            )

        return self
