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

    def act(self, state):
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, state['new_customer'][1:3])
        _, _, dual_values = perfect_hindsight(
            current_instance, state['static_info']['warehouses_location'], state['warehouses_capacity'],
            self.num_regions + 1, divisible=True
        )

        all_values = []
        for f in range(self.num_warehouses):
            if state['warehouses_capacity'][f] == 0:
                all_values.append(np.inf)
                continue
            all_values.append(distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][f]) - dual_values[f])

        return int(np.argmin(all_values))
