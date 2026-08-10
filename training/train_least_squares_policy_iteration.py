import os
import sys

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

from miscelaneous import distance_calculator

class API_Agent:
    def __init__(self, env, discount_factor=0.9, alpha=0.1, num_iterations=5, num_simulations=10, stop_criterion=1e-1):

        self.env = env
        def get_state(state):
            return state
        self.env.get_state = get_state

        self.discount_factor = discount_factor  # Gamma
        self.alpha = alpha  # Step size for parameter update
        self.num_iterations = num_iterations  # Outer loop (policy improvement)
        self.num_simulations = num_simulations  # Inner loop (policy evaluation)
        self.stop_criterion = stop_criterion  # Stopping criterion for parameter update
        
        # Initialize parameters for state-action approximation
        self.theta = np.zeros(3)  # Initialize parameters for Q-function approximation

    def extract_features(self, state, warehouse_index):
        """ Convert environment state & selected action (warehouse) into feature vector. """
        distance = distance_calculator(state['static_info']['warehouses_location'][warehouse_index], state['new_customer'][1:3])
        distance_normalized = distance / 212.13  # Normalize distance to [0,1] based on grid size
        orders_left_share = state['customers_left'] / state['static_info']['num_customers']  # Normalize orders left to [0,1]
        # relative capacity share instead of the paper's absolute c_t,f (Eq. 16): under
        # uneven initial capacity distributions, the dominant facility's absolute capacity
        # tracks orders_left almost 1:1, which destabilizes the OLS fit in fit_q_function
        # (collinear features -> unstable/flipped signs on w1/w3). The share is bounded in
        # [0,1] and decorrelated from orders_left by construction.
        capacity_share = state['warehouses_capacity'][warehouse_index] / state['static_info']['warehouses_initial_capacity'][warehouse_index]
        #capacity_share = state['warehouses_capacity'][warehouse_index] / np.sum(state['static_info']['warehouses_initial_capacity'])
        features = np.array([orders_left_share, distance_normalized, capacity_share])
        #features = np.array([orders_left, distance, state['warehouses_capacity'][warehouse_index]])
        return features

    def simulate_policy(self, policy):
        """ Run a single episode with the current policy and collect data. """
        state,_ = self.env.reset()
        all_data = []  # Stores (features, reward-to-go, action)
        final_reward = 0

        done = False
        
        while not done:
            #print('Warehouse capacities:', state['warehouses_capacity'])
            #print('Warehouses with capacity:', self.env.warehouses_with_capacity)
            #print('Waarehouses initial capacity:', self.env.warehouses_initial_capacity)
            action = policy(state, self.theta)  # Select action based on current policy
            features = self.extract_features(state, action)  # Extract action-specific features
            next_state, reward, done, truncated, info = self.env.step(action)
            final_reward += reward
        
            all_data.append((features, reward, action))
            
            state = next_state
        
        #Update all_data with the reward-to-go
        reward_to_go = 0
        for i in range(len(all_data) - 1, -1, -1):
            reward_to_go = all_data[i][1] + self.discount_factor * reward_to_go  #/(num_customers-i)
            all_data[i] = (all_data[i][0], reward_to_go, all_data[i][2])
        
        #print('All data:', all_data)
        
        return all_data, final_reward
    

    def fit_q_function(self, all_data):
        """ Update Q-function approximation parameters (theta) for each action separately. """
        theta_new = self.theta.copy()  # Copy current parameters

        # Separate data for each action
        X, y = [], []  # Data

        for features, reward, _ in all_data:
            X.append(features)
            y.append(reward)

        # Train separate regressions for each action
        model = LinearRegression().fit(X, y)
        theta_new = model.coef_  # Update only the first 3 parameters

        return theta_new

    def update_policy(self):
        """ Outer loop: Approximate Policy Iteration with action-specific updates. """
        stop_flag = False

        evolution_data = []
        all_thetas = []
        best_reward = float('-inf')  # Initialize best reward to negative infinity

        for q in range(self.num_iterations):
            if stop_flag:
                break
            print(f"Iteration {q+1}")
            all_data = []
            all_final_rewards = []

            all_thetas.append(self.theta)
            
            for _ in range(self.num_simulations):
                #data, _ = self.simulate_policy(self.greedy_policy)
                data, final_reward = self.simulate_policy(self.current_policy)
                #_, final_reward = self.simulate_policy(self.current_policy)

                all_final_rewards.append(final_reward)
                all_data.extend(data)
            
            mean_reward = np.mean(all_final_rewards)
            evolution_data.append(mean_reward)
            
            # Compute new theta using action-specific updates
            theta_new = self.fit_q_function(all_data)

            # Create a stopping criterion comparing each coefficient of theta and theta_new
            if np.all(np.abs((self.theta - theta_new) / self.theta) < self.stop_criterion):
                stop_flag = True

            # Apply exponential smoothing
            if q < self.num_iterations - 1:
                self.theta = (1 - self.alpha) * self.theta + self.alpha * theta_new

            #print(f"Iteration {q+1}: Updated Theta -> {self.theta}")
        
        # After all iterations, store a csv file with the mean reward for each iteration
        if q == self.num_iterations - 1 or stop_flag:

            print(f"Size iterations: {len(evolution_data)}")
            print(f"Size thetas: {len(range(1, q + 2))}")

            if stop_flag:
                data = {
                'iteration': list(range(1, q + 1)),
                'mean_reward': evolution_data,
                'theta': [','.join(map(str, np.round(theta, 2))) for theta in all_thetas]

                }
            else:
                data = {
                'iteration': list(range(1, q + 2)),
                'mean_reward': evolution_data,
                'theta': [','.join(map(str, np.round(theta, 2))) for theta in all_thetas]
                }
            
            file_name = f'{TRAIN_DIR}/least_squares_policy_iteration_training/lspi_evolution_w_{self.env.num_warehouses}_c_{self.env.num_customers}_d_{self.env.capacity_distribution}.csv'
            df = pd.DataFrame(data)
            df.to_csv(file_name, mode='w', header=True, index=False)

    def greedy_policy(self, state, theta):
        
        #Select the closest warehouse with capacity
        #Order warehouses by distance
        warehouses = list(enumerate(state['warehouses_capacity']))
        warehouses = sorted(warehouses, key=lambda x: distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][x[0]]))
        for warehouse_index, capacity in warehouses:
            if capacity == 0:
                continue
            else:
                best_action = warehouse_index
                break

        return best_action
    
    def current_policy(self, state, theta):
        
        """ Select the warehouse that maximizes Q-value (state-action value). """
        best_action = None
        best_q_value = float('-inf')  
        
        q_values = []
        for i in range(len(state['warehouses_capacity'])):
            if state['warehouses_capacity'][i] == 0:
                q_values.append(float('-inf'))
                continue
            features = self.extract_features(state, i)  # Get state-action features
            q_values.append(np.dot(theta, features))  # Use first 2 parameters
            
        # Select the action with the highest Q-value (if multiple, select randomly)
        best_q_value = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == best_q_value]
        best_action = min(best_actions, key=lambda x: self.extract_features(state, x)[1])
        #best_action = random.choice(best_actions)
       
        return best_action


def store_theta(num_warehouses, num_customers, capacity_distribution):
    # Initialize environment
    from env import InventoryEnv
    env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)

    # Train the API agent
    #agent = API_Agent(env, discount_factor=0.99, alpha=0.10, num_iterations=100, num_simulations=100, stop_criterion=0)
    agent = API_Agent(env, discount_factor=0.99, alpha=0.05, num_iterations=100, num_simulations=500, stop_criterion=0)
    agent.update_policy()

    data = {
    'num_warehouses': [num_warehouses],
    'num_customers': [num_customers],
    'capacity_distribution': [capacity_distribution],
    'theta': [','.join(map(str, np.round(agent.theta, 2)))]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(f'{TRAIN_DIR}/least_squares_policy_iteration_training/theta.csv', mode='a', header=not os.path.exists(f'{TRAIN_DIR}/least_squares_policy_iteration_training/theta.csv'), index=False)

num_warehouses_options = [2, 3, 4, 5]
num_customers_options = [50, 100, 200, 400]
capacity_distribution_options = ['uniform','uneven']

if __name__ == "__main__":
    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution')
                store_theta(num_warehouses, num_customers, capacity_distribution)