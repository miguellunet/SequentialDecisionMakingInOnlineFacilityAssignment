import random
from pathlib import Path

import numpy as np

from miscelaneous import distance_calculator
from policies.base_policy import BasePolicy


TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"


def read_theta(num_warehouses, num_customers, capacity_distribution):
    import pandas as pd
    import numpy as np
    # theta.csv now has a proper header row (store_theta writes one for a fresh file),
    # so read it normally instead of forcing header=None/names= - doing that would treat
    # the header row as a data row and coerce every column to string, silently breaking
    # the int/str comparisons below
    df = pd.read_csv(f'{TRAIN_DIR}/least_squares_policy_iteration_training/theta.csv')

    #Get theta columns when num_warehouses, num_customers and capacity_distribution match
    if num_customers <= 100:
        updated_num_customers = num_customers
    else:
        updated_num_customers = 100

    df = df[(df['num_warehouses'] == num_warehouses) & (df['num_customers'] == updated_num_customers) & (df['capacity_distribution'] == capacity_distribution)]
    theta = np.array(list(map(float, df['theta'].iloc[0].split(','))))
    return theta

class LeastSquaresPolicyIterationPolicy(BasePolicy):
    """Greedy w.r.t. the linear Q-function learned by LSPI: theta is loaded once (it
    doesn't depend on the episode's realized demand) and reused across resets."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):
        super().__init__(env)
        self.theta = read_theta(num_warehouses, num_customers, capacity_distribution)

    def act(self, state):
        # must mirror API_Agent.extract_features in train/train_LSPI.py exactly - theta
        # was fit against these normalized features, not raw distance/capacity/orders_left
        orders_left_share = state['customers_left'] / np.sum(state['static_info']['warehouses_initial_capacity'])

        q_values = []
        distances = []
        for i in range(len(state['warehouses_capacity'])):
            if state['warehouses_capacity'][i] == 0:
                q_values.append(float('-inf'))
                distances.append(float('inf'))
                continue
            distance = distance_calculator(state['static_info']['warehouses_location'][i], state['new_customer'][1:3])
            distances.append(distance)
            distance_normalized = distance / 212.13  # Normalize distance to [0,1] based on grid size
            capacity_share = state['warehouses_capacity'][i] / state['static_info']['warehouses_initial_capacity'][i]
            features = np.array([orders_left_share, distance_normalized, capacity_share])
            q_values.append(np.dot(self.theta, features))

        best_q_value = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == best_q_value]
        if len(best_actions) == 1:
            return best_actions[0]
        return min(best_actions, key=lambda i: distances[i])
