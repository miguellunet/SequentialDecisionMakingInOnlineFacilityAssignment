# Online Facility Assignment — SDM Tutorial

Simulation environment and policies for sequential decision-making in an online
facility assignment / inventory routing problem: customers arrive one at a time
and must be assigned to a facility (warehouse) with limited capacity.

The description of each policy and of the environment is presented in the paper.

> **Runtime warning.** Running `export_results.py` as checked in — every policy, 200
> simulation episodes per instance family, across all 32 families (`num_warehouses` in
> `[2, 3, 4, 5]` × `num_customers` in `[50, 100, 200, 400]` × `capacity_distribution` in
> `['uniform', 'uneven']`) — takes about a day and a half end to end. Full training of
> every learned policy (see `training/` below) is also slow. For a quick smoke test:
> - In `export_results.py`, lower `num_instances` (episodes per family, currently `200`)
>   and/or trim `num_warehouses_options` / `num_customers_options` /
>   `capacity_distribution_options` (currently 4 × 4 × 2 families) near the bottom of the
>   file.
> - If you're (re)training a policy, cut its training budget (generations/epochs/timesteps
>   — see the relevant `train_*` script/notebook) so you get a working artifact fast rather
>   than a paper-quality one.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`perfect_hindsight.py` uses Gurobi to solve the offline assignment problem exactly.
`gurobipy` installs fine via pip, but **solving requires a Gurobi license** — free for
academic use at https://www.gurobi.com/academia/academic-program-and-licenses/. Without
a license, Gurobi falls back to a trial mode that only solves small instances. Note that
this file isn't only used for the offline perfect-information problem — the lookahead
policies (`linear_programming_heuristic.py`, `linear_programming_exact.py`,
`point_estimate_lookahead_policy.py`, `distributional_estimate_lookahead_policy.py`,
`parameterized_lookahead_approximation_policy.py`) also call it to solve the LP over
their sampled lookahead instances.

## Quick start

**Sanity-check the environment.** `env.py` has a smoke test at the bottom that runs one
random episode and checks the paper's invariants (every order served, all capacity used):

```bash
python env.py
# OK: served 40 orders, cumulative reward -1234.56
```

**This repo is ready to run the results pipeline, but not out of the box.** `instances/`
and `training/*_training/` only keep their folder structure in version control (the
actual instance files and trained model artifacts are gitignored — they're large/
regenerable, see [`.gitignore`](.gitignore)). Before any of the notebooks/scripts below
will run, you need to:

1. **Generate the instances.** Run `python instances_generator.py` from the repo root —
   it populates `instances/instances_test/`, `instances_train_imitation_learning/` and
   `instances_train_parameterized_lookahead_approximation/`. It seeds `numpy` per
   instance file so results stay reproducible.
2. **Train every policy that needs offline training.** Run each script/notebook listed in
   the `training/` section below (`train_genetic_programming.py`,
   `train_linear_value_function_approximation.py`, `train_imitation_learning.ipynb`,
   `train_parameterized_lookahead_approximation.ipynb`, `train_deep_q_networks.ipynb`,
   `train_proximal_policy_optimization.ipynb`, `train_exact_value_function.py`). Each one
   writes its artifacts into its own `training/<policy>_training/` subfolder.

Once instances and trained artifacts exist, reproduce the paper's tables and figures with:

1. `export_results.py` — runs every policy on the held-out test instances
   (`instances/instances_test/`) and writes the raw per-run numbers to `results/tables/`
   and `results/full_tables/`. This is the slow step (it actually executes each policy).
2. `export_tables.ipynb` — reads `results/full_tables/` and produces the LaTeX tables
   used in the paper (policy comparison, LVFA/PLA parameter tables).
3. `export_plots_results.ipynb` — turns the same result tables into the paper's results
   figures (cost/runtime scatter, proximity-rank bar charts), saved under `results/plots/`.
4. `export_plots_training.ipynb` — turns the raw `training/*_training/` logs into the
   paper's training-curve figures (GP/DQN/LVFA/PLA), saved under `results/plots/`.

Run them from the repo root and roughly in that order — each downstream notebook reads
files the previous ones write. All of them add the repo root to `sys.path`, so
`import env`, `import miscelaneous`, `from policies...` work regardless of which
notebook/script is running.

## Layout

Core simulation code, shared by every policy and every notebook below:

- `env.py` — the `InventoryEnv` gym environment (facility locations/capacities, customer
  arrivals, reward). Run directly (`python env.py`) for a smoke test.
- `miscelaneous.py` — small shared helpers (distance calculation, `read_instance`).
- `perfect_hindsight.py` — offline/hindsight-optimal baseline, solved exactly with
  Gurobi; also used as a training signal by the imitation-learning policy and as the LP
  solver called by the lookahead policies (see Setup above).
- `instances_generator.py` — generates the customer-arrival instances (`instances_seed_*.json`)
  used across every experiment.
- `visualize.py` — plotting/animation helpers used to produce the standalone figures in
  `visualization_examples/` (not part of the results pipeline in `results/`).
- `export_results.py`, `export_tables.ipynb`, `export_plots_results.ipynb`, `export_plots_training.ipynb` — the results pipeline
  described in Quick start above.
- `requirements.txt` — pinned dependencies for `pip install -r requirements.txt`.

### `instances/`

Generated customer-arrival instances, one JSON file per random seed
(`instances_seed_<n>.json`), each a pool of 1000 orders (individual runs just take the
first `num_customers` of that pool). Three subfolders:

- `instances_test/` — held-out instances used for evaluation (`export_results.py`).
- `instances_train_imitation_learning/` — instances used to train the imitation-learning
  policy.
- `instances_train_parameterized_lookahead_approximation/` — instances used to tune the
  parameterized lookahead approximation (PLA) policy.

### `policies/`

One file per policy, all implementing the common `BasePolicy` interface
(`base_policy.py`: `reset(instance)` once per episode, `act(state)` once per order).
Examples: `myopic_policy.py`, `random_policy.py`, `genetic_programming_policy.py`,
`linear_value_function_approximation_policy.py`, `deep_q_networks_policy.py`,
`proximal_policy_optimization_policy.py`, `imitation_learning_policy.py`,
`point_estimate_lookahead_policy.py`, `distributional_estimate_lookahead_policy.py`,
`parameterized_lookahead_approximation_policy.py`, `linear_programming_heuristic.py`,
`linear_programming_exact.py`, `perfect_hindsight_policy.py`, `exact_value_function_policy.py`.

`auxiliaries/` holds code shared by more than one policy file, factored out instead of
duplicated: `direct_lookahead_functions.py` (region-generation helpers used by the
lookahead/LP-based policies) and `imitation_learning_model.py` (the PyTorch network used
by the imitation-learning policy).

### `results/`

Output of the evaluation pipeline (`export_results.py` → `export_tables.ipynb` /
`export_plots_results.ipynb` / `export_plots_training.ipynb`):

- `tables/` and `full_tables/` — per-configuration CSVs (`table_w_<warehouses>_c_<customers>_d_<distribution>.csv`);
  `full_tables/` additionally keeps the raw per-episode rewards/times/proximity ranks
  that the plots are built from.
- `plots/` — rendered figures (PNG/PDF).

### `training/`

Where each policy that needs offline training is trained, and where its raw training
artifacts are stored:

- `train_genetic_programming.py` → `genetic_programming_training/`
- `train_linear_value_function_approximation.py` → `linear_value_function_approximation_training/`
- `train_imitation_learning.ipynb` → `imitation_learning_training/`
- `train_parameterized_lookahead_approximation.ipynb` → `parameterized_lookahead_approximation_training/`
- `train_deep_q_networks.ipynb` → `deep_q_networks_training/`
- `train_proximal_policy_optimization.ipynb` → `proximal_policy_optimization_training/`
- `train_exact_value_function.py` → `exact_value_function_training/` — unlike the others,
  this isn't a learned model: `bellman()` computes the post-decision value function
  exactly via backward induction (customer location discretized into `num_squares` grid
  cells to take the expectation) and the script stores it as one CSV of
  `capacity -> value` per `(num_warehouses, num_customers, capacity_distribution)`
  combination, read back by `exact_value_function_policy.py`.

Each script/notebook is only needed if you want to regenerate its policy's trained
artifacts from scratch — the results pipeline in `results/` reads these artifacts as-is.

### `visualization_examples/`

Standalone illustrative figures used in the paper, produced by the functions in
`visualize.py` (`warehouse_assignments.png`, `warehouse_simulation.gif`, `facilities.pdf`,
`myopic_vs_lookahead.pdf`). Independent of the `results/` pipeline — these illustrate how
the environment/policies behave rather than report performance numbers.
