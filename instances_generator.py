import numpy as np
import json

def create_instances(seed, subfolder=''):

    # generates one long pool of 1000 orders per seed; individual experiments only need
    # |T| in {50, 100, 200, 400} (Table 6) and take instance[:num_customers], so every
    # smaller run is a prefix of the larger runs from the same seed
    num_customers = 1000

    np.random.seed(seed)

    instances = []
    for i in range(num_customers):
        # [id, x, y, demand] - same shape as env.py's create_customer(). The (-100, 100)
        # bounds mirror env.py's default grid_size=200 but are NOT read from it, so keep
        # them in sync manually if grid_size ever changes
        customer = []
        customer.append(i+1)
        customer.append(np.random.uniform(-100, 100))
        customer.append(np.random.uniform(-100, 100))
        customer.append(1)
        instances.append(customer)

    # Store in json format
    instances_json = {}
    instances_json['instances'] = instances

    #Store in json file
    with open(f'instances/{subfolder}instances_seed_{seed}.json', 'w') as f:
        json.dump(instances_json, f)

if __name__ == "__main__":

    for seed in range(10001, 11001):
        create_instances(seed, subfolder='instances_test/')

    for seed in range(20001, 25001):
        create_instances(seed, subfolder='instances_train_imitation_learning/')

    for seed in range(30001, 30501):
            create_instances(seed, subfolder='instances_train_parameterized_lookahead_approximation/')