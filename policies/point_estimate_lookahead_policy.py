
import numpy as np

from miscelaneous import distance_calculator
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy


class PointEstimateLookaheadPolicy(BasePolicy):

    def __init__(self, env, dla_param=1, num_samples=10, seed=42):
        super().__init__(env)
        self.dla_param = dla_param
        self.num_samples = num_samples
        # dedicated RNG (not the global np.random) so this policy's Monte Carlo draws
        # are reproducible on their own, regardless of what else has consumed
        # np.random's global state before or after this instance is created
        self.rng = np.random.default_rng(seed)

        # first perfect_hindsight() call pays Gurobi's one-time environment/license
        # checkout cost; warm it up here so it doesn't land inside the first timed
        # act() call. Use a synthetic full-capacity state instead of self.env.obs -
        # self.env is shared across policies (see run_episode in export_results.py),
        # so its obs could be a just-finished, exhausted episode. Note act() also reads
        # self.env.warehouses_capacity directly (not from state), so this only guards
        # against a stale 'new_customer'/'customers_left'; a depleted self.env would
        # still need a fresh env to avoid an infeasible LP here.
        warmup_state = {'customers_left': self.env.num_customers, 'new_customer': [0, 0.0, 0.0, 1]}
        self.act(warmup_state)

    def act(self, state):
        new_customer = state['new_customer']
        customers_left = state['customers_left']
        updated_capacity = [int(cap * self.dla_param) for cap in self.env.warehouses_capacity]      #dla_param is set to 1 but can be changed to scale the capacity of the warehouses in the sampled instance

        votes = []
        for _ in range(self.num_samples):
            # customer 0 is the real, already-known customer; the rest are a random
            # sample of where the remaining (customers_left - 1) orders will land
            instance = [[0, new_customer[1], new_customer[2], 1]]
            for i in range(customers_left - 1):
                x = self.rng.uniform(-self.env.grid_size / 2, self.env.grid_size / 2)
                y = self.rng.uniform(-self.env.grid_size / 2, self.env.grid_size / 2)
                instance.append([i + 1, x, y, 1])

            assignments, _, _ = perfect_hindsight(
                instance, self.env.warehouses_location, updated_capacity, len(instance)
            )
            votes.append(assignments[0])

        #print(f"Votes: {votes}")

        max_count = max(votes.count(f) for f in set(votes))
        tied_facilities = [f for f in set(votes) if votes.count(f) == max_count]
        return min(tied_facilities, key=lambda f: distance_calculator(self.env.warehouses_location[f], new_customer[1:3]))
