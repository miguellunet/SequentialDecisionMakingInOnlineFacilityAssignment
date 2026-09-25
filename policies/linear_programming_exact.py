import numpy as np

from miscelaneous import distance_calculator
from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy

class LinearProgrammingExactPolicy(BasePolicy):

    def __init__(self, env, num_warehouses, num_regions=100):
        super().__init__(env)
        self.num_warehouses = num_warehouses
        self.num_regions = num_regions

        warmup_state = {
            'customers_left': self.env.num_customers,
            'new_customer': [0, 0.0, 0.0, 1],
            'warehouses_capacity': self.env.warehouses_initial_capacity.copy(),
            'static_info': {'warehouses_location': self.env.warehouses_location},
        }
        self.act(warmup_state)

    def act(self, state):
        
        current_instance = generate_regions_instance(state['customers_left']-1, self.num_regions, self.env.grid_size, None)

        all_values = []
        for f in range(self.num_warehouses):
            if state['warehouses_capacity'][f] == 0:
                all_values.append(np.inf)
                continue
            new_capacity = state['warehouses_capacity'].copy()
            new_capacity[f] -= 1
            _, obj, _ = perfect_hindsight(
                current_instance, state['static_info']['warehouses_location'], new_capacity,
                self.num_regions, divisible=True
            )
            #new_lambda = baseline_obj - obj
            new_lambda = obj
            all_values.append(distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][f]) + new_lambda)

        return int(np.argmin(all_values))