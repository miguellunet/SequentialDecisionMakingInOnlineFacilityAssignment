from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Common interface every policy must implement so run_episode can stay agnostic
    to which policy it's driving - it just calls reset() once per episode and act()
    once per step, regardless of whether the policy needs to carry state between
    those calls (e.g. DLAPolicy's synthetic instance, PerfectHindsightPolicy's
    precomputed action sequence) or is stateless (e.g. DistancePolicy)."""

    def __init__(self, env):
        self.env = env

    def reset(self, instance):
        """Called once at the start of each episode, before any act() calls, with the
        realized order sequence for that episode. Also responsible for pointing
        env.get_state at whatever transform this policy's act() expects - default is
        the identity (raw state dict), since running one policy right after another on
        the same env would otherwise leak the previous policy's get_state override
        (e.g. DQNPolicy's flattened-vector transform) into this one. Override to
        (re)initialize any other per-episode state; call super().reset(instance) first
        if the identity default is still wanted."""
        self.env.get_state = lambda state: state

    @abstractmethod
    def act(self, state):
        """Return the warehouse index to serve state['new_customer'] with."""
        raise NotImplementedError
