import numpy as np
import torch
from pathlib import Path

from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

class ProximalPolicyOptimizationPolicy(BasePolicy):
  
    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):
        super().__init__(env)
        from sb3_contrib import MaskablePPO  # deferred: only needed when this policy is actually instantiated

        self.model = MaskablePPO.load(
            TRAIN_DIR / "proximal_policy_optimization_training" / "rl_models" / "ppo_models" / f"ppo_model_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}"
        )

        # first predict() calls pay PyTorch's one-time thread-pool/backend init cost plus
        # sb3-contrib's masking/obs-conversion setup; warm up with 10 calls here so it
        # doesn't land inside the first timed act() call
        warmup_size = 1
        dummy_obs = np.zeros(2 * num_warehouses, dtype=np.float32)
        dummy_mask = np.ones(num_warehouses, dtype=bool)
        for _ in range(warmup_size):
            self.model.predict(dummy_obs, action_masks=dummy_mask, deterministic=True)

    def _rl_get_state(self, state):
        data_rows = []
        for i in range(state['static_info']['num_warehouses']):
            data_rows.append(state['warehouses_distance'][i] / 212.13)
            data_rows.append(state['warehouses_capacity'][i] / state['static_info']['warehouses_initial_capacity'][i])

        return np.array(np.nan_to_num(data_rows, nan=0), dtype=np.float32)

    def reset(self, instance):
        self.env.get_state = self._rl_get_state

    def act(self, state):
        action_masks = np.array([capacity > 0 for capacity in self.env.warehouses_capacity])
        action, _ = self.model.predict(state, action_masks=action_masks, deterministic=True)
        return int(action)
