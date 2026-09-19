import numpy as np
import torch
from stable_baselines3 import DQN
from pathlib import Path

from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

class DeepQNetworksPolicy(BasePolicy):

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):

        # Load the trained DQN model from the specified path based on the number of warehouses, customers, and capacity distribution.
        super().__init__(env)
        self.model = DQN.load(
            TRAIN_DIR / "deep_q_networks_training" / "rl_models" / "dqn_models" / f"dqn_model_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}"
        )

        warmup_size = 1
        # first forward passes pay PyTorch's one-time thread-pool/backend init cost
        # Warm it up here so it doesn't land inside the first timed act() call.
        dummy_state = np.zeros(2 * num_warehouses, dtype=np.float32)
        for _ in range(warmup_size):
            self.act(dummy_state)

    def _rl_get_state(self, state):

        # Flatten the state for the RL model
        # Distance is normalized by the maximum distance (212.13), and capacity is normalized by the initial capacity

        data_rows = []
        for i in range(state['static_info']['num_warehouses']):
            data_rows.append(state['warehouses_distance'][i] / 212.13)
            data_rows.append(state['warehouses_capacity'][i] / state['static_info']['warehouses_initial_capacity'][i])

        return np.array(np.nan_to_num(data_rows, nan=0), dtype=np.float32)

    def reset(self, instance):
        self.env.get_state = self._rl_get_state

    def act(self, state):

        # Only able to serve customers if the warehouse has capacity, so mask out warehouses with no capacity

        q_values = self.model.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()
        q_values = q_values[0]
        masked_q_values = [q if cap > 0 else -np.inf for q, cap in zip(q_values, self.env.warehouses_capacity)]
        return int(np.argmax(masked_q_values))