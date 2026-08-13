import random

from policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """Picks uniformly at random among the warehouses that still have capacity."""

    def __init__(self, env):
        super().__init__(env)

        # warm-up call, kept consistent with the other policies' init-time act() call
        # even though this one has no real first-call cost to hide
        self.act(self.env.obs)

    def act(self, state):
        available = [i for i in range(self.env.num_warehouses) if state['warehouses_capacity'][i] > 0]
        return random.choice(available)
