from pathlib import Path
import pandas as pd
import numpy as np

from miscelaneous import distance_calculator
from policies.base_policy import BasePolicy


TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"


def read_theta(num_warehouses, num_customers, capacity_distribution):
    
    df = pd.read_csv(f'{TRAIN_DIR}/linear_value_function_approximation_training/theta.csv')

    df = df[(df['num_warehouses'] == num_warehouses) & (df['num_customers'] == num_customers) & (df['capacity_distribution'] == capacity_distribution)]
    # theta.csv is appended to on every training run (see
    # train_linear_value_function_approximation.py) - iloc[-1] takes the most recent
    # run for this family, not the first one ever recorded.
    theta = np.array(list(map(float, df['theta'].iloc[-1].split(','))))
    return theta


class LeastSquaresPolicyIterationImprovedPolicy(BasePolicy):
    """Greedy w.r.t. a linear state-value function V(s) = theta . warehouses_capacity
    (raw capacity per facility, no intercept, fit by LSPIImprovedAgent - see
    train_linear_value_function_approximation.py). V is fit on reward-to-go (reward =
    -dist, Eq. 9) so it's a reward-based (negative-cost) value; the facility is chosen
    minimizing dist(s,f) - gamma * V(s after assigning to f); theta doesn't depend on
    the episode's realized demand, so it's loaded once and reused across resets.

    If include_squared_capacity_feature is True, V(s) gets a second feature per
    facility (capacity_i^2, alongside capacity_i), each with its own coefficient -
    not a single feature shared across facilities. theta must then have
    2 * num_warehouses entries: the first num_warehouses weight capacity_i, the
    next num_warehouses weight capacity_i^2 (same facility order in both blocks)."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution, discount_factor=0.99, include_squared_capacity_feature=False):
        super().__init__(env)
        self.theta = read_theta(num_warehouses, num_customers, capacity_distribution)
        self.discount_factor = discount_factor
        self.include_squared_capacity_feature = include_squared_capacity_feature

        # warm-up calls, kept consistent with the other policies' init-time act() calls
        # even though this one has no real first-call cost to hide; 10 calls so any
        # init-time dynamics settle before the first timed act() call
        warmup_size = 1
        for _ in range(warmup_size):
            self.act(self.env.obs)

    def act(self, state):
        capacities = np.array(state['warehouses_capacity'], dtype=float)

        costs = []
        distances = []
        for i in range(len(capacities)):
            if capacities[i] == 0:
                costs.append(float('inf'))
                distances.append(float('inf'))
                continue
            distance = distance_calculator(state['static_info']['warehouses_location'][i], state['new_customer'][1:3])
            distances.append(distance)
            next_capacities = capacities.copy()
            next_capacities[i] -= 1
            if self.include_squared_capacity_feature:
                features = np.concatenate([next_capacities, next_capacities ** 2])
            else:
                features = next_capacities
            next_value = np.dot(self.theta, features)
            costs.append(distance + self.discount_factor * next_value)
        
        best_cost = min(costs)
        best_actions = [i for i, c in enumerate(costs) if c == best_cost]
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return min(best_actions, key=lambda i: distances[i])