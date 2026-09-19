import numpy as np

from miscelaneous import distance_calculator
from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy


class LinearProgrammingHeuristicPolicy(BasePolicy):
    """Solves the region-approximated LP relaxation for the current state to get
    opportunity-cost dual values per warehouse (Eqs. 25-28), then picks the warehouse
    minimizing distance minus that dual value."""

    def __init__(self, env, num_warehouses, num_regions=100):
        super().__init__(env)
        self.num_warehouses = num_warehouses
        self.num_regions = num_regions

        # first perfect_hindsight() call pays Gurobi's one-time environment/license
        # checkout cost; warm it up here so it doesn't land inside the first timed
        # act() call. Use a synthetic full-capacity state instead of self.env.obs -
        # self.env is shared across policies (see run_episode in export_results.py),
        # so its obs could be a just-finished, exhausted episode.
        warmup_state = {
            'customers_left': self.env.num_customers,
            'new_customer': [0, 0.0, 0.0, 1],
            'warehouses_capacity': self.env.warehouses_initial_capacity.copy(),
            'static_info': {'warehouses_location': self.env.warehouses_location},
        }
        self.act(warmup_state)

    def act(self, state):
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, None)
        _, _, dual_values = perfect_hindsight(
            current_instance, state['static_info']['warehouses_location'], state['warehouses_capacity'],
            self.num_regions, divisible=True
        )

        all_values = []
        for f in range(self.num_warehouses):
            if state['warehouses_capacity'][f] == 0:
                all_values.append(np.inf)
                continue
            all_values.append(distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][f]) - dual_values[f])

        return int(np.argmin(all_values))
