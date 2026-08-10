import numpy as np
import torch
from pathlib import Path

from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"


class ProximalPolicyOptimizationPolicy(BasePolicy):
    """Wraps an sb3-contrib MaskablePPO model trained on a flattened observation (not
    the raw state dict the other policies use) - reset() points env.get_state at the
    same flattening function DQNPolicy uses (train_ppo() and train_dqn() build the env
    the same way, without overriding get_state), so act() receives that vector. Unlike
    DQN, MaskablePPO takes the capacity mask directly as a predict() argument rather
    than needing it baked into a custom network, since depleted warehouses are only
    invalid, not literally absent from the action space."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):
        super().__init__(env)
        from sb3_contrib import MaskablePPO  # deferred: only needed when this policy is actually instantiated

        self.model = MaskablePPO.load(
            TRAIN_DIR / "proximal_policy_optimization_training" / "rl_models" / "ppo_models" / f"ppo_model_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}"
        )

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
