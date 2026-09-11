import numpy as np

from miscelaneous import distance_calculator
from policies.base_policy import BasePolicy
from training.train_exact_value_function import aggregate_state, read_bellman_values


class ExactValueFunctionPolicy(BasePolicy):
    """Greedy policy w.r.t. the post-decision value function V(c) solved exactly by
    bellman() (training/train_exact_value_function.py) - c is the capacity remaining per facility, with
    customer location never part of the state (only entering the immediate reward).
    V is read from the table bellman() stored on an aggregation-rounded grid (see
    aggregate_state); the facility chosen maximizes
    -dist(customer, f) + gamma * V(aggregate(capacity with f -= 1))."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution, gamma=1, num_squares=10_000, aggregation=1):
        super().__init__(env)
        self.values = read_bellman_values(num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
        self.gamma = gamma
        self.aggregation = aggregation

        # warm-up call, kept consistent with the other policies' init-time act() call
        self.act(self.env.obs)

    def act(self, state):
        capacities = np.array(state['warehouses_capacity'], dtype=int)

        rewards = []
        distances = []
        for i in range(len(capacities)):
            if capacities[i] == 0:
                rewards.append(float('-inf'))
                distances.append(float('inf'))
                continue
            distance = distance_calculator(state['static_info']['warehouses_location'][i], state['new_customer'][1:3])
            distances.append(distance)
            next_capacities = capacities.copy()
            next_capacities[i] -= 1
            next_value = self.values.get(aggregate_state(tuple(next_capacities), self.aggregation), 0.0)
            rewards.append(-distance + self.gamma * next_value)

        best_reward = max(rewards)
        best_actions = [i for i, r in enumerate(rewards) if r == best_reward]
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return min(best_actions, key=lambda i: distances[i])
