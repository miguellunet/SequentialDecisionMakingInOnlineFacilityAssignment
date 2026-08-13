
from policies.auxiliaries.direct_lookahead_functions import generate_dla_instance, update_instance_location
from perfect_hindsight import perfect_hindsight
from policies.base_policy import BasePolicy


class PointEstimateLookaheadPolicy(BasePolicy):
    """Deterministic Look-Ahead: approximates the future demand it can't see with a
    synthetic grid of evenly-spaced points (see generate_dla_instance), swaps the real
    current customer into that grid each step, and re-solves the assignment LP so the
    decision accounts for an approximation of what's still to come."""

    def __init__(self, env, dla_param=1):
        super().__init__(env)
        self.dla_param = dla_param

        # first perfect_hindsight() call pays Gurobi's one-time environment/license
        # checkout cost; warm it up here so it doesn't land inside the first timed
        # act() call. dla_instance is normally built in reset() per episode, but
        # act() needs one to run at all, so build a throwaway one here too - it gets
        # overwritten by the real reset() before the first real episode.
        self.dla_instance = generate_dla_instance(self.env.num_customers, self.env.grid_size)
        self.act(self.env.obs)

    def reset(self, instance):
        super().reset(instance)
        self.dla_instance = generate_dla_instance(self.env.num_customers, self.env.grid_size)

    def act(self, state):
        instance_pre_action, instance_post_action = update_instance_location(self.dla_instance, state['new_customer'][1:3])     #Removes the closest point to the new customer from the dla instance and adds the new customer to the dla instance
        updated_capacity = [int(cap * self.dla_param) for cap in self.env.warehouses_capacity]      #dla_param is set to 1 but can be changed to scale the capacity of the warehouses in the dla instance
        assignments, _, _ = perfect_hindsight(
            instance_pre_action, self.env.warehouses_location, updated_capacity, len(instance_pre_action)
        )
        self.dla_instance = instance_post_action    #Removes the new customer from the dla instance
        return assignments[0]