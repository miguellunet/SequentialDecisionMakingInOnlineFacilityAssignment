
# Library importation
import os
import sys
import csv
import itertools
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import defaultdict

import numpy as np
import pandas as pd

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

from env import InventoryEnv
from miscelaneous import distance_calculator


# Below this many states, a backward-induction level is processed inline in the
# parent instead of being dispatched to the worker pool - dispatch (pickling +
# IPC round trip per chunk) costs more than it saves for small levels.
MIN_STATES_FOR_PARALLEL = 200


def _dp_worker_init(shm_name, shape, distances, gamma, all_possible_actions):
    # runs once per worker process (Pool initializer), not per task/level - attaches
    # to the parent's shared-memory value table so every worker reads/writes the
    # same array with no per-level re-sync or serialization of the (potentially huge)
    # value table.
    global _shm, _V, _distances, _gamma, _all_possible_actions
    _shm = shared_memory.SharedMemory(name=shm_name)
    _V = np.ndarray(shape, dtype=np.float64, buffer=_shm.buf)
    _distances = distances
    _gamma = gamma
    _all_possible_actions = all_possible_actions


def _dp_process_chunk(chunk):
    # writes each state's value directly into the shared V array - no return value
    # needs pickling back to the parent, unlike a plain multiprocessing.Pool passing
    # dict fragments around.
    V = _V
    distances = _distances
    gamma = _gamma
    for state in chunk:
        possible_actions = [a for a in _all_possible_actions if state[a] > 0]
        next_values = np.empty(len(possible_actions))
        for i, action in enumerate(possible_actions):
            next_state = list(state)
            next_state[action] -= 1
            next_values[i] = V[tuple(next_state)]
        scores = -distances[:, possible_actions] + gamma * next_values
        V[state] = scores.max(axis=1).mean()


def _chunkify(states, n_chunks):
    n_chunks = max(1, n_chunks)
    chunk_size = max(1, (len(states) + n_chunks - 1) // n_chunks)
    return [states[i:i + chunk_size] for i in range(0, len(states), chunk_size)]


def aggregate_state(capacity, aggregation):
    """Round each facility's remaining capacity UP to the nearest multiple of
    `aggregation` (e.g. aggregation=5: 3 -> 5, 11 -> 15). aggregation=1 is the
    identity (exact per-unit granularity)."""
    if aggregation <= 1:
        return tuple(capacity)
    return tuple(int(np.ceil(c / aggregation)) * aggregation for c in capacity)


def backward_dynamic_programming(env, gamma=0.99, num_squares=16, aggregation=1, n_workers=None):
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

    aggregation=1 (the only value any current production config uses) runs on
    _dp_parallel: real capacity states map 1:1 onto a dense array, so the
    value table can live in shared memory and be split across worker processes
    with zero approximation - every value comes out bit-identical to the plain
    sequential loop, just computed with more cores. aggregation>1 falls back to
    _dp_sequential, since its cross-state bucket-averaging doesn't map onto
    a dense per-state array as cleanly.
    """
    if aggregation <= 1:
        return _dp_parallel(env, gamma=gamma, num_squares=num_squares, n_workers=n_workers)
    return _dp_sequential(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)



def _dp_parallel(env, gamma=0.99, num_squares=16, n_workers=None):
    n_workers = n_workers or os.cpu_count() or 1
    print(f"Running dynamic backward induction with {n_workers} worker processes")

    all_possible_actions = np.arange(0, env.num_warehouses, 1)

    grid_side = round(num_squares ** 0.5)
    if grid_side * grid_side != num_squares:
        raise ValueError(f"num_squares={num_squares} must be a perfect square")

    cell_width = env.grid_size / grid_side
    half_grid = env.grid_size / 2
    centers = [-half_grid + cell_width * (i + 0.5) for i in range(grid_side)]
    possible_locations = [(x, y) for x in centers for y in centers]

    distances = np.array([
        [distance_calculator(loc, facility_loc) for facility_loc in env.warehouses_location]
        for loc in possible_locations
    ])

    caps = env.warehouses_capacity
    shape = tuple(c + 1 for c in caps)

    states_by_sum = defaultdict(list)
    for state in itertools.product(*(range(0, c + 1) for c in caps)):
        states_by_sum[sum(state)].append(state)
    total_capacity = sum(caps)

    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 8)
    try:
        V = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
        V[:] = 0.0

        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers, initializer=_dp_worker_init,
                      initargs=(shm.name, shape, distances, gamma, all_possible_actions)) as pool:
            for customers_left in range(1, total_capacity + 1):
                print(f"Dynamic programming iteration for {customers_left} customers left ({total_capacity - customers_left} done)")

                states_list = states_by_sum[customers_left]
                if not states_list:
                    continue

                if len(states_list) < MIN_STATES_FOR_PARALLEL:
                    # dispatch overhead would exceed the work itself - run inline
                    for state in states_list:
                        possible_actions = [a for a in all_possible_actions if state[a] > 0]
                        next_values = np.empty(len(possible_actions))
                        for i, action in enumerate(possible_actions):
                            next_state = list(state)
                            next_state[action] -= 1
                            next_values[i] = V[tuple(next_state)]
                        scores = -distances[:, possible_actions] + gamma * next_values
                        V[state] = scores.max(axis=1).mean()
                else:
                    pool.map(_dp_process_chunk, _chunkify(states_list, n_workers))

        values = {state: float(V[state]) for state in itertools.product(*(range(0, c + 1) for c in caps))}
    finally:
        shm.close()
        shm.unlink()

    return values



def _dp_sequential(env, gamma=0.99, num_squares=16, aggregation=1):
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

    # distance(loc, facility) never depends on the state c, only on the fixed grid
    # and facility locations - precompute it once instead of recomputing it for
    # every state visited during backward induction.
    distances = np.array([
        [distance_calculator(loc, facility_loc) for facility_loc in env.warehouses_location]
        for loc in possible_locations
    ])

    states_by_sum = defaultdict(list)
    for state in itertools.product(*(range(0, capacity + 1) for capacity in env.warehouses_capacity)):
        states_by_sum[sum(state)].append(state)

    total_capacity = sum(env.warehouses_capacity)

    values = {aggregate_state((0,) * env.num_warehouses, aggregation): 0.0}

    for customers_left in range(1, total_capacity + 1):
        print(f"Dynamic programming iteration for {customers_left} customers left ({total_capacity - customers_left} done)")

        generation_values = defaultdict(list)

        for state in states_by_sum[customers_left]:

            possible_actions = [action for action in all_possible_actions if state[action] > 0]

            next_values = np.empty(len(possible_actions))
            for i, action in enumerate(possible_actions):
                next_state = list(state)
                next_state[action] -= 1
                next_key = aggregate_state(tuple(next_state), aggregation)
                next_values[i] = values.get(next_key, 0.0)

            # E_L[ max_a(...) ] vectorized over all locations at once: for every grid
            # cell, -distance + gamma*V(next) per feasible action, then max over
            # actions, then average over locations - replaces a per-location Python
            # loop (with a per-location max() call) with two numpy reductions.
            scores = -distances[:, possible_actions] + gamma * next_values
            state_value = scores.max(axis=1).mean()

            generation_values[aggregate_state(state, aggregation)].append(state_value)

        for key, vals in generation_values.items():
            values[key] = float(np.mean(vals))

    return values


def store_dp_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares=16, aggregation=1):
    info_dir = os.path.join(TRAIN_DIR, 'exact_value_function_training')
    os.makedirs(info_dir, exist_ok=True)
    path = os.path.join(info_dir, f'values_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_gamma_{gamma}_squares_{num_squares}_agg_{aggregation}.csv')
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['capacity', 'value'])
        for capacity, value in values.items():
            writer.writerow([','.join(map(str, capacity)), value])


def read_dp_values(num_warehouses, num_customers, capacity_distribution, gamma, num_squares=16, aggregation=1):
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
num_warehouses_options = [5]
num_customers_options = [200]
capacity_distribution_options = ['uniform', 'uneven']
#capacity_distribution_options = ['uneven']
gamma = 1
num_squares = 10_000
aggregation = 1

if __name__ == "__main__":

    '''
    num_warehouses_options = [2, 3, 4, 5]
    num_customers_options = [50]
    capacity_distribution_options = ['uniform', 'uneven']

    NUM_REGIONS_TEST = [25, 100, 400, 1600, 2500, 5625, 10000, 22500, 40000]

    #Store the value of the first state of the value function for each configuration of the parameters
    df = pd.DataFrame(columns=['num_warehouses', 'num_customers', 'capacity_distribution', 'num_squares', 'initial_value'])
    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                for num_squares in NUM_REGIONS_TEST:
                    print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution, {num_squares} squares')
                    env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)


                    values = _dp_sequential(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)
                    store_dp_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
                    initial_value = values.get(aggregate_state(tuple(env.warehouses_initial_capacity), aggregation), 0.0)
                    df = pd.concat([df, pd.DataFrame.from_records([{
                        'num_warehouses': num_warehouses,
                        'num_customers': num_customers,
                        'capacity_distribution': capacity_distribution,
                        'num_squares': num_squares,
                        'initial_value': initial_value
                    }])], ignore_index=True)
    #Store the value of the first state of the value function for each configuration of the parameters
    df.to_csv(os.path.join(TRAIN_DIR, 'exact_value_function_training', 'square_test.csv'), index=False, float_format='%.5f')
    '''
                    
    
    num_customers_options = [50, 100, 200, 400]
    num_warehouses_options = [2, 3, 4, 5]
    capacity_distribution_options = ['uniform', 'uneven']
    
    training_times = []  # one row per (num_warehouses, num_customers, capacity_distribution) family

    gamma = 1
    num_squares = 10_000 #10_000
    aggregation = 1

    
    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                if num_warehouses == 5 and num_customers == 400:
                    continue  # skip this combination because it takes too long to run
                print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution')
                env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)

                start_time = time.perf_counter()
                values = backward_dynamic_programming(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)
                training_time_minutes = (time.perf_counter() - start_time) / 60

                store_dp_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
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