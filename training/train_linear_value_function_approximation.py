import os
import sys

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from miscelaneous import distance_calculator


class LVFAAgent:
    """Fits a linear state-value function V(s) = theta . warehouses_capacity (raw,
    non-normalized, no intercept - so V(all-zero capacity) = 0 by construction) via
    Monte Carlo rollout targets, then acts greedily w.r.t. the one-step Bellman
    recursion: since reward = -dist (Eq. 9), V(s) is a reward-based (i.e. negative-
    cost) value, so the greedy choice maximizes -dist(s,f) + gamma * V(s after
    assigning to f). Unlike
    least_squares_policy_iteration (which fits a per-(state,action) Q-function on
    designed/normalized features), here the value function only depends on the state
    - the customer's location only enters through the immediate cost term dist(s,f),
    since demand is i.i.d. uniform and carries no information about future cost.

    If include_squared_capacity_feature is True, V(s) gets a second feature per
    facility (capacity_i^2, alongside capacity_i), each with its own coefficient -
    not a single feature shared across facilities. This keeps the per-facility
    additive structure that current_policy's cost(i) comparison relies on: changing
    facility i's capacity only ever moves facility i's own two terms."""

    def __init__(self, env, discount_factor=0.9, alpha=0.1, num_iterations=5, num_simulations=10, stop_criterion=1e-1, include_squared_capacity_feature=False):
        self.env = env

        def get_state(state):
            return state
        self.env.get_state = get_state

        self.discount_factor = discount_factor
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_simulations = num_simulations
        self.stop_criterion = stop_criterion
        self.include_squared_capacity_feature = include_squared_capacity_feature

        # one coefficient per facility, no intercept, plus one more per facility for
        # the squared-capacity term if enabled
        self.theta = np.zeros(env.num_warehouses * (2 if include_squared_capacity_feature else 1))

    def features_of(self, capacities):
        """ Feature vector for a (post- or pre-decision) capacity vector: the raw
        capacities, plus - if enabled - each facility's squared capacity appended
        as a second, separately-weighted block. """
        if not self.include_squared_capacity_feature:
            return capacities
        else:
            return np.concatenate([capacities, capacities ** 2])

    def value_of(self, theta, features):
        return float(np.dot(theta, features))

    def simulate_policy(self, policy):
        """ Run a single episode with the current policy and collect data. """
        state, _ = self.env.reset()
        all_data = []  # Stores (features_at_t, reward)
        final_reward = 0

        done = False
        while not done:
            action = policy(state, self.theta)
            # feature vector for this decision epoch is built from the pre-decision
            # capacities - V doesn't depend on which action is taken, only on the state
            capacities = np.array(state['warehouses_capacity'], dtype=float)
            features = self.features_of(capacities)
            next_state, reward, done, truncated, info = self.env.step(action)
            final_reward += reward
        
            all_data.append((features, reward))
            state = next_state

        # Update all_data with the (discounted) reward-to-go
        reward_to_go = 0
        for i in range(len(all_data) - 1, -1, -1):
            reward_to_go = all_data[i][1] + self.discount_factor * reward_to_go
            all_data[i] = (all_data[i][0], reward_to_go)

        return all_data, final_reward

    def fit_v_function(self, all_data):
        """ Update V-function approximation parameters (theta) via OLS, no intercept. """
        X, y = [], []
        for features, reward_to_go in all_data:
            X.append(features)
            y.append(reward_to_go)

        model = LinearRegression(fit_intercept=False).fit(X, y)
        #model = LinearRegression.fit(X, y)
        return model.coef_

    def update_policy(self):
        """ Outer loop: Approximate Policy Iteration with a state-value update. """
        stop_flag = False

        evolution_data = []
        all_thetas = []

        for q in range(self.num_iterations):
            if stop_flag:
                break
            print(f"Iteration {q+1}")
            all_data = []
            all_final_rewards = []

            all_thetas.append(self.theta)

            for _ in range(self.num_simulations):
                data, final_reward = self.simulate_policy(self.current_policy)
                all_final_rewards.append(final_reward)
                all_data.extend(data)

            mean_reward = np.mean(all_final_rewards)
            evolution_data.append(mean_reward)

            theta_new = self.fit_v_function(all_data)

            # Stopping criterion comparing each coefficient of theta and theta_new
            # (skip warehouses that never held nonzero capacity during fitting - div by 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_change = np.abs((self.theta - theta_new) / self.theta)
            rel_change = rel_change[np.isfinite(rel_change)]
            if rel_change.size > 0 and np.all(rel_change < self.stop_criterion):
                stop_flag = True

            if q < self.num_iterations - 1:
                self.theta = (1 - self.alpha) * self.theta + self.alpha * theta_new

        if q == self.num_iterations - 1 or stop_flag:
            print(f"Size iterations: {len(evolution_data)}")
            print(f"Size thetas: {len(range(1, q + 2))}")

            if stop_flag:
                data = {
                    'iteration': list(range(1, q + 1)),
                    'mean_reward': evolution_data,
                    'theta': [','.join(map(str, np.round(theta, 4))) for theta in all_thetas]
                }
            else:
                data = {
                    'iteration': list(range(1, q + 2)),
                    'mean_reward': evolution_data,
                    'theta': [','.join(map(str, np.round(theta, 4))) for theta in all_thetas]
                }

            file_name = f'{TRAIN_DIR}/linear_value_function_approximation_training/lvfa_improved_evolution_w_{self.env.num_warehouses}_c_{self.env.num_customers}_d_{self.env.capacity_distribution}.csv'
            df = pd.DataFrame(data)
            df.to_csv(file_name, mode='w', header=True, index=False)

    def current_policy(self, state, theta):
        """ Select the warehouse f maximizing -dist(s,f) + gamma * V(state after choosing f)
        (equivalently: minimizing dist(s,f) - gamma * V(...)). V is fit on reward-to-go
        (reward = -dist), so it's a reward-based value and enters with a plus sign here. """
        capacities = np.array(state['warehouses_capacity'], dtype=float)

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
            next_features = self.features_of(next_capacities)
            rewards.append(-distance + self.discount_factor * self.value_of(theta, next_features))

        best_reward = max(rewards)
        best_actions = [i for i, r in enumerate(rewards) if r == best_reward]
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return min(best_actions, key=lambda i: distances[i])


def store_theta(num_warehouses, num_customers, capacity_distribution, include_squared_capacity_feature=False):
    from env import InventoryEnv
    env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)

    agent = LVFAAgent(env, discount_factor=0.99, alpha=0.01, num_iterations=500, num_simulations=100, stop_criterion=0, include_squared_capacity_feature=include_squared_capacity_feature)
    agent.update_policy()

    data = {
        'num_warehouses': [num_warehouses],
        'num_customers': [num_customers],
        'capacity_distribution': [capacity_distribution],
        'theta': [','.join(map(str, np.round(agent.theta, 4)))]
    }

    df = pd.DataFrame(data)
    theta_path = f'{TRAIN_DIR}/linear_value_function_approximation_training/theta.csv'
    df.to_csv(theta_path, mode='a', header=not os.path.exists(theta_path), index=False)


num_warehouses_options = [2, 3, 4, 5]
num_customers_options = [50, 100, 200, 400]
capacity_distribution_options = ['uniform', 'uneven']

if __name__ == "__main__":
    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution')
                store_theta(num_warehouses, num_customers, capacity_distribution, False)
