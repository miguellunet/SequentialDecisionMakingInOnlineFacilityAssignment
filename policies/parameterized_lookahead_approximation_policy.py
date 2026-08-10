import pandas as pd
from pathlib import Path

from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"


class ParameterizedLookaheadApproximationPolicy(BasePolicy):
    """DLARegionsPolicy with each warehouse's remaining capacity scaled by a tuned
    "best_param" (a cost-function-approximation correction, calibrated per
    num_warehouses/num_customers/capacity_distribution and looked up once here)."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution, num_regions=100, param=None):
        super().__init__(env)
        self.num_regions = num_regions
        if param is not None:
            # Lets train_cfadla.ipynb's parameter sweep reuse this class directly for
            # each candidate value, instead of duplicating act()'s logic inline.
            self.param = param
        else:
            cfa_params_df = pd.read_csv(TRAIN_DIR / "parameterized_lookahead_approximation_training" / "parameterized_lookahead_approximation_best_params.csv")
            row = cfa_params_df[
                (cfa_params_df['num_warehouses'] == num_warehouses)
                & (cfa_params_df['num_customers'] == num_customers)
                & (cfa_params_df['capacity_distribution'] == capacity_distribution)
            ]
            self.param = row['best_param'].values[0]

    def act(self, state):
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, state['new_customer'][1:3])
        warehouses_capacity = state['warehouses_capacity'].copy()
        for i in range(len(warehouses_capacity)):
            warehouses_capacity[i] = warehouses_capacity[i] * self.param
        assignments, _, _ = perfect_hindsight(
            current_instance, state['static_info']['warehouses_location'], warehouses_capacity,
            self.num_regions + 1, divisible=True, divisible_except_first_one=True
        ) #the +1 is because the new customer is added to the instance, so we have num_regions + 1 customers in total
        return assignments[0]
