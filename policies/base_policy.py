from abc import ABC, abstractmethod

class BasePolicy(ABC):
    
    # Common policy architecture for all policies. All policies should inherit from this class and implement the act() method.
    
    def __init__(self, env):
        self.env = env

    def reset(self, instance):
        # Called once at the start of each episode, before any act() calls

        self.env.get_state = lambda state: state

    @abstractmethod
    def act(self, state):
        # Return the warehouse index to serve state['new_customer'] with.
        raise NotImplementedError
