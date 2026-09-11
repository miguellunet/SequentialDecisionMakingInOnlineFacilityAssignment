
# Library importation
import os
import sys
import csv
import itertools
import time
from collections import defaultdict

import numpy as np
import pandas as pd

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

from env import InventoryEnv
from miscelaneous import distance_calculator


def aggregate_state(capacity, aggregation):
    """Round each facility's remaining capacity UP to the nearest multiple of
    `aggregation` (e.g. aggregation=5: 3 -> 5, 11 -> 15). aggregation=1 is the
    identity (exact per-unit granularity)."""
    if aggregation <= 1:
        return tuple(capacity)
    return tuple(int(np.ceil(c / aggregation)) * aggregation for c in capacity)


def bellman(env, gamma=0.99, num_squares=16, aggregation=1):
    """Backward induction on the post-decision value function V(c), where c is the
    remaining capacity per facility - customer location is never part of the state,
    it only enters the immediate reward. For a post-decision state c (capacity right
    after a decision, before the next customer's location is known):

        V(c) = E_L[ max_{f : c_f>0} ( -dist(L,f) + gamma * V(c with c_f -= 1) ) ]

    L (the arriving customer's location) is discretized into `num_squares` grid
    cells purely to take this expectation - the expectation of the max is taken
    per capacity state, not the max of the expectation, so each grid cell still
    picks its own best facility before averaging.

    `aggregation` controls the granularity of the table V is stored on (see
    aggregate_state): the recursion itself always steps through every real capacity
    vector one unit at a time (a decision depletes exactly one unit of the chosen
    facility's real capacity), but reads/writes go through the aggregated key, so
    every real state sharing that key contributes to (and can be overwritten by) the
    same table entry - aggregation=1 keeps this exact.
    """

    all_possible_actions = np.arange(0, env.num_warehouses, 1)

    # num_squares must be a perfect square k^2: the grid_size x grid_size customer
    # area is split into a k-by-k lattice of equal cells, each cell represented by
    # its center point (verified to reproduce the old hardcoded 4/16-square grids)
    grid_side = round(num_squares ** 0.5)
    if grid_side * grid_side != num_squares:
        raise ValueError(f"num_squares={num_squares} must be a perfect square")

    cell_width = env.grid_size / grid_side
    half_grid = env.grid_size / 2
    centers = [-half_grid + cell_width * (i + 0.5) for i in range(grid_side)]

    possible_locations = [(x, y) for x in centers for y in centers]
    f = 1 / num_squares

    states_by_sum = defaultdict(list)
    for state in itertools.product(*(range(0, capacity + 1) for capacity in env.warehouses_capacity)):
        states_by_sum[sum(state)].append(state)

    total_capacity = sum(env.warehouses_capacity)

    values = {aggregate_state((0,) * env.num_warehouses, aggregation): 0.0}

    for customers_left in range(1, total_capacity + 1):
        #print(f"Bellman iteration for {customers_left} customers left ({total_capacity - customers_left} done)")

        generation_values = defaultdict(list)

        for state in states_by_sum[customers_left]:

            possible_actions = [action for action in all_possible_actions if state[action] > 0]

            next_value_by_action = {}
            for action in possible_actions:
                next_state = list(state)
                next_state[action] -= 1
                next_key = aggregate_state(tuple(next_state), aggregation)
                next_value_by_action[action] = values.get(next_key, 0.0)

            state_value = 0.0
            for loc in possible_locations:
                best = max(
                    -distance_calculator(loc, env.warehouses_location[action]) + gamma * next_value_by_action[action]
                    for action in possible_actions
                )
                state_value += f * best

            generation_values[aggregate_state(state, aggregation)].append(state_value)

        for key, vals in generation_values.items():
            values[key] = float(np.mean(vals))

    return values


def store_bellman_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares=16, aggregation=1):
    info_dir = os.path.join(TRAIN_DIR, 'exact_value_function_training')
    os.makedirs(info_dir, exist_ok=True)
    path = os.path.join(info_dir, f'values_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_gamma_{gamma}_squares_{num_squares}_agg_{aggregation}.csv')
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['capacity', 'value'])
        for capacity, value in values.items():
            writer.writerow([','.join(map(str, capacity)), value])


def read_bellman_values(num_warehouses, num_customers, capacity_distribution, gamma, num_squares=16, aggregation=1):
    path = os.path.join(TRAIN_DIR, 'exact_value_function_training', f'values_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_gamma_{gamma}_squares_{num_squares}_agg_{aggregation}.csv')
    values = {}
    with open(path, 'r') as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            capacity = tuple(int(x) for x in row[0].split(','))
            values[capacity] = float(row[1])
    return values


num_warehouses_options = [2, 3, 4, 5]
num_customers_options = [50]
capacity_distribution_options = ['uniform', 'uneven']
gamma = 1
num_squares = 10_000
aggregation = 1

if __name__ == "__main__":
    training_times = []  # one row per (num_warehouses, num_customers, capacity_distribution) family

    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution')
                env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)

                start_time = time.perf_counter()
                values = bellman(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)
                training_time_minutes = (time.perf_counter() - start_time) / 60

                store_bellman_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
                num_states = 1
                for capacity in env.warehouses_initial_capacity:
                    num_states = num_states * (capacity + 1)
                training_times.append({
                    'num_warehouses': num_warehouses,
                    'num_customers': num_customers,
                    'capacity_distribution': capacity_distribution,
                    'training_time': round(training_time_minutes, 2),
                    'states': num_states
                })

    info_dir = os.path.join(TRAIN_DIR, 'exact_value_function_training')
    os.makedirs(info_dir, exist_ok=True)

    training_times_df = pd.DataFrame(training_times)
    training_times_df.to_csv(os.path.join(info_dir, 'exact_value_function_training_times.csv'), index=False, float_format='%.5f')