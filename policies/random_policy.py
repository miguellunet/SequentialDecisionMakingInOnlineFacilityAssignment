import random

from policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """Picks uniformly at random among the warehouses that still have capacity."""

    def act(self, state):
        available = [i for i in range(self.env.num_warehouses) if state['warehouses_capacity'][i] > 0]
        return random.choice(available)
