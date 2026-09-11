import pandas as pd
from pathlib import Path

from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mult(a, b): return a * b
def _div(a, b): return b and a / b or 0


class GeneticProgrammingPolicy(BasePolicy):
    """Evaluates the best genetic-programming expression found for this
    (num_warehouses, num_customers, capacity_distribution) combo, per warehouse, and
    picks the highest-scoring one that still has capacity."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):
        super().__init__(env)
        df = pd.read_csv(TRAIN_DIR / "genetic_programming_training" / "ALL_TIME_best_individual.csv")
        df = df[
            (df['num_warehouses'] == num_warehouses)
            & (df['num_customers'] == num_customers)
            & (df['capacity_distribution'] == capacity_distribution)
        ]
        # ALL_TIME_best_individual.csv is appended to on every training run (see
        # train_genetic_programming.py) - iloc[-1] takes the most recent run for this
        # family, not the first one ever recorded.
        expression = df.iloc[-1]['best_ind']
        self.compiled_expression = compile(expression, '<string>', 'eval')

        # warm-up call, kept consistent with the other policies' init-time act() call
        # even though this one has no real first-call cost to hide
        self.act(self.env.obs)

    def act(self, state):
        add, sub, mult, div = _add, _sub, _mult, _div

        # Only score warehouses that still have capacity - an evolved expression is an
        # unconstrained combination of add/sub/mult/div and can overflow to inf/-inf/NaN,
        # which would tie with (or beat) a -inf sentinel used to mask out empty warehouses
        # and let argmax pick one anyway. Excluding them from the candidate set entirely
        # avoids that regardless of what the expression computes for the valid ones.
        scores = {}
        for i in range(len(state['warehouses_capacity'])):
            if state['warehouses_capacity'][i] == 0:
                continue

            DISTANCE = state['warehouses_distance'][i]
            CAPACITY = state['warehouses_capacity'][i]
            INITIAL_CAPACITY = state['static_info']['warehouses_initial_capacity'][i]
            scores[i] = eval(self.compiled_expression)

        best_score = max(scores.values())
        tied = [i for i, score in scores.items() if score == best_score]
        action = min(tied, key=lambda i: state['warehouses_distance'][i])

        return action
