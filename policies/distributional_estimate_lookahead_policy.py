from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy


class DistributionalEstimateLookaheadPolicy(BasePolicy):
    """Like DLAPolicy, but approximates future demand with num_regions aggregated
    region-points (via generate_regions_instance) instead of one point per customer -
    cheaper to re-solve each step for larger instances."""

    def __init__(self, env, num_regions=100):
        super().__init__(env)
        self.num_regions = num_regions

    def act(self, state):
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, state['new_customer'][1:3])
        assignments, _, _ = perfect_hindsight(
            current_instance, state['static_info']['warehouses_location'], state['warehouses_capacity'],
            self.num_regions + 1, divisible=True, divisible_except_first_one=True
        ) #The +1 is because the new customer is added to the instance, so we have num_regions + 1 customers in total
        return assignments[0]
