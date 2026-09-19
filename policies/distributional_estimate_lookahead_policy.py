from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy

class DistributionalEstimateLookaheadPolicy(BasePolicy):

    def __init__(self, env, num_regions=100):
        super().__init__(env)
        self.num_regions = num_regions

        # first perfect_hindsight() call pays Gurobi's one-time environment/license checkout cost
        warmup_state = {
            'customers_left': self.env.num_customers,
            'new_customer': [0, 0.0, 0.0, 1],
            'warehouses_capacity': self.env.warehouses_initial_capacity.copy(),
            'static_info': {'warehouses_location': self.env.warehouses_location},
        }
        self.act(warmup_state)

    def act(self, state):

        #Generates a lookahead instance with the new customer and the expected remaining customers, and solves it with deterministic optimization to get the best assignment for the new customer.
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, state['new_customer'][1:3])

        assignments, _, _ = perfect_hindsight(
            current_instance, state['static_info']['warehouses_location'], state['warehouses_capacity'],
            self.num_regions + 1, divisible=True, divisible_except_first_one=True
        ) #The +1 is because the new customer is added to the instance, so we have num_regions + 1 customers in total
        return assignments[0]
