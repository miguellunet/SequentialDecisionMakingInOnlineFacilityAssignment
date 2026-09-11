import numpy as np
from gymnasium import Env, spaces
from miscelaneous import distance_calculator


class InventoryEnv(Env):

    def __init__ (self, num_warehouses, num_customers, capacity_distribution, grid_size = 200):
        
        self.num_warehouses = num_warehouses
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.capacity_distribution = capacity_distribution
        self.booked_customers = 0
        self.instance = None

        if num_warehouses == 2:
            self.warehouses_location = [[-50, -50], [50, 50]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.5*num_customers), int(0.5*num_customers)]
                # int() truncation can undercount total capacity; top up the first few
                # facilities so sum(capacity) == num_customers exactly (every order must
                # be served - Section 3, Q2)
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.7*num_customers), int(0.3*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 3:
            self.warehouses_location = [[-50, -50], [0, 50], [50, -50]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(num_customers/3), int(num_customers/3), int(num_customers/3)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.5*num_customers), int(0.3*num_customers), int(0.2*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 4:
            self.warehouses_location = [[-50, -50], [-50, 50], [50, 50], [50, -50]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.25*num_customers) for i in range(num_warehouses)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.4*num_customers), int(0.3*num_customers), int(0.2*num_customers), int(0.1*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 5:
            self.warehouses_location = [[-50, -50], [-50, 50], [50, 50], [50, -50], [0, 0]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.20*num_customers) for i in range(num_warehouses)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.3*num_customers), int(0.25*num_customers), int(0.20*num_customers), int(0.15*num_customers), int(0.10*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 10:
            # extra config kept for exploration beyond the paper - Table 6 only reports
            # results for 2-5 facilities
            self.warehouses_location = [[-50, -60], [-50, 80], [50, 20], [50, -30], [0, 0], [-40, 0], [50, 0], [80, -80], [0, 90], [-80, -70]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.1*num_customers) for i in range(num_warehouses)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.2*num_customers), int(0.15*num_customers), int(0.15*num_customers), int(0.1*num_customers), int(0.1*num_customers), int(0.1*num_customers), int(0.05*num_customers), int(0.07*num_customers), int(0.03*num_customers), int(0.05*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        
        self.warehouses_proximity = []
        #Store a vector of vectors, in which each vector is the rank of the warehouse in terms of proximity to the other warehouses
        # consumed by policies that need a spillover/priority order over warehouses (e.g. train_gp.py)
        for i in range(num_warehouses):
            distances = [distance_calculator(self.warehouses_location[i], self.warehouses_location[j]) for j in range(num_warehouses)]
            new_rank = np.argsort(distances)
            #Remove the first element, which is the warehouse itself
            new_rank = new_rank[1:]
            self.warehouses_proximity.append(new_rank)
            

        self.warehouses_capacity = self.warehouses_initial_capacity.copy()
        # the eligible action set A_t from Eq. (6): facilities with remaining capacity
        self.warehouses_with_capacity = [i for i in range(self.num_warehouses)]
        self.seed = 0
        np.random.seed(self.seed)

        if self.instance == None:
            self.new_customer = self.create_customer()
        else:
            self.new_customer = self.instance[self.booked_customers]
        
        self.obs = self.reset()

        obs_dim = len(self.reset()[0])
        #print('Obervation Dimension:', obs_dim)
        low_dim = np.array([0 for i in range(obs_dim)])
        high_dim = np.array([1 for i in range(obs_dim)])
        self.observation_space = spaces.Box(low = low_dim, high = high_dim, dtype = np.float32)
        self.action_space = spaces.Discrete(self.num_warehouses)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        
    def refresh_seed(self):
        np.random.seed(self.seed)

    def set_instance(self, instance):
        self.instance = instance

    def create_customer(self):
        id = self.booked_customers + 1
        x = np.random.uniform(-self.grid_size/2, self.grid_size/2)
        y = np.random.uniform(-self.grid_size/2, self.grid_size/2)
        demand = 1
        return [id, x, y, demand]

    def step(self, action):

        #Calculate the reward
        reward = 0
        done = False
        truncated = False
        info = {}

        if action not in self.warehouses_with_capacity:
            # a policy should never select a facility outside A_t (Eq. 6) - raise instead
            # of silently rerouting so bugs in the calling policy surface immediately
            raise ValueError(f"Action {action} selects a warehouse with 0 capacity remaining (warehouses_with_capacity={self.warehouses_with_capacity})")

        warehouse = action
        self.warehouses_capacity[action] -= 1
        if self.warehouses_capacity[action] <= 0:
            self.warehouses_with_capacity.remove(warehouse)
        
        self.all_customers.append(self.new_customer)
        self.all_warehouses.append(warehouse)
        self.all_capacities.append(self.warehouses_capacity.copy())

        # R(S_t, a_t) = -d_{t,a_t}, Eq. (9) - undiscounted, no cost-per-km scaling
        reward -= distance_calculator(self.warehouses_location[warehouse], self.new_customer[1:3])
        self.booked_customers += 1

        if len(self.warehouses_with_capacity) == 0:
            done = True
        else:
            if self.instance == None:
                self.new_customer = self.create_customer()
            else:
                self.new_customer = self.instance[self.booked_customers]

        obs = {'static_info': {'warehouses_initial_capacity': self.warehouses_initial_capacity, 'warehouses_location': self.warehouses_location, 'num_warehouses': self.num_warehouses, 'num_customers': self.num_customers, 'warehouses_proximity': self.warehouses_proximity},
            'warehouses_capacity': self.warehouses_capacity, 'new_customer': self.new_customer, 'warehouses_distance': [distance_calculator(self.warehouses_location[i], self.new_customer[1:3]) for i in range(self.num_warehouses)], 'booked_customers': self.booked_customers, 'customers_left': self.num_customers - self.booked_customers}

        self.obs = obs
        
        state = self.get_state(self.obs)

        return state, reward, done, truncated, info

    def get_facility_proximity(self, action):
        # Given the action, calculate the rank among warehouses with positive capacity only.
        available_indices = [i for i in range(self.num_warehouses) if self.warehouses_capacity[i] > 0]
        distances = [distance_calculator(self.warehouses_location[i], self.new_customer[1:3]) for i in available_indices]
        sorted_available = np.array(available_indices)[np.argsort(distances)]
        proximity_rank = np.where(sorted_available == action)[0][0] + 1  # +1 to make it 1-indexed
        return proximity_rank
    
    def get_state(self, state):
        # default flat-vector observation (used as-is by DQN via stable-baselines3).
        # Other policies (e.g. LSPI) monkey-patch this method with their own feature
        # set - see train_LSPI.py - so changes here don't affect them.

        data_rows = []

        for i in range(self.num_warehouses):
            # NOTE: max corner-to-corner distance (~283) exceeds grid_size (200), so
            # distant facilities saturate to 1.0 after the clip below and become
            # indistinguishable to DQN
            data_rows.append(state['warehouses_distance'][i]/212.13)
            data_rows.append(state['warehouses_capacity'][i]/state['static_info']['warehouses_initial_capacity'][i])

        #Truncate the data to be between 0 and 1
        #data_rows = np.clip(data_rows, 0, 1)
        
        return np.array(np.nan_to_num(data_rows, nan=0), dtype=np.float32)
            
    def reset(self, seed=0, options=None):

        self.warehouses_capacity = self.warehouses_initial_capacity.copy()
        self.warehouses_with_capacity = [i for i in range(self.num_warehouses)]
        self.booked_customers = 0

        if self.instance == None:
            self.new_customer = self.create_customer()
        else:
            self.new_customer = self.instance[self.booked_customers]
    
        self.all_customers = []
        self.all_warehouses = []
        self.all_capacities = []

        obs = {'static_info': {'warehouses_initial_capacity': self.warehouses_initial_capacity, 'warehouses_location': self.warehouses_location, 'num_warehouses': self.num_warehouses, 'num_customers': self.num_customers,'warehouses_proximity': self.warehouses_proximity},
            'warehouses_capacity': self.warehouses_capacity, 'new_customer': self.new_customer, 'warehouses_distance': [distance_calculator(self.warehouses_location[i], self.new_customer[1:3]) for i in range(self.num_warehouses)], 'booked_customers': self.booked_customers, 'customers_left': self.num_customers - self.booked_customers,
            'warehouses_initial_capacity': self.warehouses_initial_capacity, 'warehouses_location': self.warehouses_location}

        self.obs = obs
        
        state = self.get_state(self.obs)

        return state, {}
    

if __name__ == "__main__":
    # smoke test: run one random episode end-to-end and sanity-check the invariants
    # the paper assumes (Section 3, Q2: every order served, capacity exhausted exactly)
    env = InventoryEnv(3, 40, 'uniform', grid_size=100)

    def get_state(state):
        return state
    env.get_state = get_state

    state, _ = env.reset()

    done = False
    cum_reward = 0

    while not done:
        action = np.random.randint(0, state['static_info']['num_warehouses'])
        state, reward, done, truncated, info = env.step(action)
        cum_reward += reward

    assert env.booked_customers == env.num_customers, "not all orders were served"
    assert sum(env.warehouses_capacity) == 0, "capacity left unused at episode end"
    print(f"OK: served {env.booked_customers} orders, cumulative reward {cum_reward:.2f}")