from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy


class PerfectHindsightPolicy(BasePolicy):
    """Solves the whole episode's assignment optimally upfront - which needs the full
    order sequence the other policies never get to see - and just replays the
    resulting actions step by step. Serves as the theoretical upper bound."""

    def __init__(self, env):
        super().__init__(env)

        # first perfect_hindsight() call pays Gurobi's one-time environment/license
        # checkout cost; warm it up here (with a throwaway single-customer instance,
        # since the real order sequence isn't known until reset()) so it doesn't land
        # inside the first timed act() call
        perfect_hindsight([self.env.new_customer], self.env.warehouses_location, self.env.warehouses_initial_capacity, 1)

    def reset(self, instance):
        super().reset(instance)
        self.all_actions, _, _ = perfect_hindsight(
            instance, self.env.warehouses_location, self.env.warehouses_initial_capacity, self.env.num_customers
        )
        self.idx = 0

    def act(self, state):
        action = self.all_actions[self.idx]
        self.idx += 1
        return action
