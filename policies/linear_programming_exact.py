import numpy as np

from miscelaneous import distance_calculator
from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy


class LinearProgrammingExactPolicy(BasePolicy):
    """Like AcimovicPolicy, but instead of LP dual values, re-solves the region LP once
    per candidate warehouse with its capacity bumped by one unit, and uses the resulting
    objective improvement as that warehouse's marginal value (more expensive, exact
    version of the same opportunity-cost idea)."""

    def __init__(self, env, num_warehouses, num_regions=100):
        super().__init__(env)
        self.num_warehouses = num_warehouses
        self.num_regions = num_regions

    def act(self, state):
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, state['new_customer'][1:3])
        _, baseline_obj, _ = perfect_hindsight(
            current_instance, state['static_info']['warehouses_location'], state['warehouses_capacity'],
            self.num_regions + 1, divisible=True
        )

        all_values = []
        for f in range(self.num_warehouses):
            if state['warehouses_capacity'][f] == 0:
                all_values.append(np.inf)
                continue
            new_capacity = state['warehouses_capacity'].copy()
            new_capacity[f] += 1
            _, obj, _ = perfect_hindsight(
                current_instance, state['static_info']['warehouses_location'], new_capacity,
                self.num_regions + 1, divisible=True
            )
            new_lambda = baseline_obj - obj
            all_values.append(distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][f]) + new_lambda)

        return int(np.argmin(all_values))
