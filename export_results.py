# Library imports
import numpy as np
import pandas as pd
import os
import time

# File imports
from env import InventoryEnv
from miscelaneous import read_instance

from policies import linear_programming_exact
from policies.random_policy import RandomPolicy
from policies.myopic_policy import MyopicPolicy
from policies.genetic_programming_policy import GeneticProgrammingPolicy
from policies.linear_value_function_approximation_policy import LeastSquaresPolicyIterationImprovedPolicy
from policies.deep_q_networks_policy import DeepQNetworksPolicy
from policies.proximal_policy_optimization_policy import ProximalPolicyOptimizationPolicy
from policies.imitation_learning_policy import ImitationLearningPolicy
from policies.point_estimate_lookahead_policy import PointEstimateLookaheadPolicy
from policies.distributional_estimate_lookahead_policy import DistributionalEstimateLookaheadPolicy
from policies.parameterized_lookahead_approximation_policy import ParameterizedLookaheadApproximationPolicy
from policies.linear_programming_heuristic import LinearProgrammingHeuristicPolicy
from policies.linear_programming_exact import LinearProgrammingExactPolicy
from policies.perfect_hindsight_policy import PerfectHindsightPolicy
from policies.exact_value_function_policy import ExactValueFunctionPolicy


def run_episode(env, policy, instance_file):

    instance = read_instance(instance_file)
    env.set_instance(instance)
    policy.reset(instance)  # let the policy configure env.get_state and any per-episode state before env.reset()
    state, info = env.reset()

    done = False
    final_reward = 0
    time_per_order_list = []
    facility_proximty_list = []

    while not done:
        start_time = time.time()
        action = policy.act(state)
        time_per_order_list.append(time.time() - start_time)

        #Given the action, calculate if is the closets, 2nd closest, n closts... to the customer
        facility_proximty_list.append(env.get_facility_proximity(action))

        state, reward, done, truncated, _ = env.step(action)
        final_reward += reward

    return final_reward, time_per_order_list, facility_proximty_list

def evaluate_policy(env, policy, instances):

    list_all_rewards = []
    list_all_times = []
    list_all_facility_proximty = []

    for i in range(len(instances)):

        final_reward, time_per_order_list, facility_proximty_list = run_episode(env, policy, instances[i])

        list_all_times.append(time_per_order_list)
        list_all_rewards.append(final_reward)
        list_all_facility_proximty.append(facility_proximty_list)

    return np.mean(list_all_rewards), np.std(list_all_rewards), np.mean(list_all_times), np.std(list_all_times), list_all_rewards, list_all_times, list_all_facility_proximty


def export_results(num_warehouses, num_customers, capacity_distribution):

    env = InventoryEnv(num_warehouses, num_customers, capacity_distribution, grid_size = 200)

    num_instances = 200
    instances = [f'instances/instances_test/instances_seed_{10000+i}.json' for i in range(1,num_instances+1)]

    
    if num_customers == 400 and num_warehouses == 5:
        
        policies = {
            'perfect_hindsight': PerfectHindsightPolicy(env),
            'imitation_learning': ImitationLearningPolicy(env, num_warehouses, num_customers, capacity_distribution),
            'myopic': MyopicPolicy(env),
            'genetic_programming': GeneticProgrammingPolicy(env, num_warehouses, num_customers, capacity_distribution),
            'linear_value_function_approximation': LeastSquaresPolicyIterationImprovedPolicy(env, num_warehouses, num_customers, capacity_distribution),
            'deep_q_networks': DeepQNetworksPolicy(env, num_warehouses, num_customers, capacity_distribution),
            'point_estimate_lookahead': PointEstimateLookaheadPolicy(env),
            'distributional_estimate_lookahead': DistributionalEstimateLookaheadPolicy(env),
            'linear_programming_heuristic': LinearProgrammingHeuristicPolicy(env, num_warehouses),
            'linear_programming_exact': LinearProgrammingExactPolicy(env, num_warehouses),
            'parameterized_lookahead_approximation': ParameterizedLookaheadApproximationPolicy(env, num_warehouses, num_customers, capacity_distribution),
            'proximal_policy_optimization': ProximalPolicyOptimizationPolicy(env, num_warehouses, num_customers, capacity_distribution)
        }

    else:

        policies = {
                    'perfect_hindsight': PerfectHindsightPolicy(env),
                    'imitation_learning': ImitationLearningPolicy(env, num_warehouses, num_customers, capacity_distribution),
                    'myopic': MyopicPolicy(env),
                    'genetic_programming': GeneticProgrammingPolicy(env, num_warehouses, num_customers, capacity_distribution),
                    'linear_value_function_approximation': LeastSquaresPolicyIterationImprovedPolicy(env, num_warehouses, num_customers, capacity_distribution),
                    'deep_q_networks': DeepQNetworksPolicy(env, num_warehouses, num_customers, capacity_distribution),
                    'point_estimate_lookahead': PointEstimateLookaheadPolicy(env),
                    'distributional_estimate_lookahead': DistributionalEstimateLookaheadPolicy(env),
                    'linear_programming_heuristic': LinearProgrammingHeuristicPolicy(env, num_warehouses),
                    'linear_programming_exact': LinearProgrammingExactPolicy(env, num_warehouses),
                    'parameterized_lookahead_approximation': ParameterizedLookaheadApproximationPolicy(env, num_warehouses, num_customers, capacity_distribution),
                    'proximal_policy_optimization': ProximalPolicyOptimizationPolicy(env, num_warehouses, num_customers, capacity_distribution),
                    'exact_value_function': ExactValueFunctionPolicy(env, num_warehouses, num_customers, capacity_distribution)
        }

    
    results = []
    full_results = []

    for policy_name, policy in policies.items():

        print(f"Evaluating policy: {policy_name} for {num_warehouses} warehouses, {num_customers} customers, capacity distribution: {capacity_distribution}")

        mean_reward, std_reward, mean_time, std_time, all_rewards, all_times, all_facility_proximty = evaluate_policy(env, policy, instances)

        results.append({
            'Policy': policy_name,
            'MeanReward': np.round(mean_reward,2),
            'StdReward': np.round(std_reward,2),
            'MeanTime': np.round(mean_time*1_000_000_000,6),
            'StdTime': np.round(std_time*1_000_000_000,6)
        })

        full_results.append({
                    'Policy': policy_name,
                    'MeanReward': np.round(mean_reward,2),
                    'StdReward': np.round(std_reward,2),
                    'MeanTime': np.round(mean_time*1_000_000_000,6),
                    'StdTime': np.round(std_time*1_000_000_000,6),
                    'AllRewards': [round(float(r), 2) for r in all_rewards],
                    'AllTimes': [[round(float(t*1_000_000_000), 6) for t in times] for times in all_times],
                    'AllFacilityProximity': all_facility_proximty
                })

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    full_results_df = pd.DataFrame(full_results)

    # Save the results to a CSV file
    #results_df.to_csv(f'results/tables/table_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv', mode='a', header=not os.path.exists(f'results/tables/table_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv'), index=False)
    #full_results_df.to_csv(f'results/full_tables/full_results_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv', mode='a', header=not os.path.exists(f'results/full_tables/full_results_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv'), index=False)
    results_df.to_csv(f'results/tables/table_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv', mode='a', header=not os.path.exists(f'results/tables/table_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv'), index=False)
    full_results_df.to_csv(f'results/full_tables/full_results_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv', mode='a', header=not os.path.exists(f'results/full_tables/full_results_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}.csv'), index=False)
    


num_warehouses_options = [2, 3, 4, 5]
num_customers_options = [50, 100, 200, 400]
capacity_distribution_options = ['uniform', 'uneven']

for num_customers in num_customers_options:
    for num_warehouses in num_warehouses_options:
        for capacity_distribution in capacity_distribution_options:
            print(f"Running experiments for {num_warehouses} warehouses, {num_customers} customers and {capacity_distribution} capacity distribution")
            export_results(num_warehouses, num_customers, capacity_distribution)