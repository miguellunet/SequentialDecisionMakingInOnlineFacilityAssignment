from miscelaneous import distance_calculator
from gurobipy import GRB, Model
import numpy as np

# Fixes Gurobi's own internal tie-breaking/search order, so a MIP with multiple
# equally-optimal solutions (e.g. the binary assignment PHS and PEL solve) returns the
# same one every run instead of depending on thread-scheduling nondeterminism.
GUROBI_SEED = 42

# Cap the solve at 1 second so large instances don't blow up runtime, but never return
# empty-handed: if no incumbent was found by the cutoff, keep solving - now with no
# time limit but capped to the first solution found - so the caller always gets an
# assignment (optimal, or the best incumbent at 1s, or the first feasible solution
# found after 1s).
#TIME_LIMIT_SECONDS = 1

def perfect_hindsight(instance, warehouses_location, warehouses_initial_capacity, num_customers, divisible = False, divisible_except_first_one = False):

    # despite the name, this one function is the shared MIP engine behind several
    # policies in the paper, selected via (divisible, divisible_except_first_one):
    #   (False, *)        -> binary assignment, Eqs. (1)-(4): the PHS baseline, and the
    #                        per-step model solved by PEL (Section 5.4.1)
    #   (True,  False)     -> fully continuous LP, used to derive the opportunity-cost
    #                        dual values for LPH/LPE (Eqs. 25-28, Section 5.5.2)
    #   (True,  True)      -> order 0 forced binary, the rest continuous: mirrors the
    #                        zf (binary) + ur,f (continuous) structure of DEL/PLA
    #                        (Eqs. 17-22 and 23)

    #print('Total number of customers:', num_customers)
    #print('Warehouses inventory:', warehouses_initial_capacity)
    #print('Total demand:', sum([customer[3] for customer in instance[:num_customers]]))
    
    all_customers = instance[:num_customers]

    # Create a MIP model to assign customers to warehouses

    # Sets
    T = np.arange(len(all_customers)) # Set of customers
    F = np.arange(len(warehouses_initial_capacity)) # Set of warehouses

    # Parameters
    d = np.array([[distance_calculator(customer[1:3], warehouses_location[f]) for f in F] for customer in all_customers])   # distance from customer t to warehouse f
    c = warehouses_initial_capacity     # capacity of warehouse f
    s = np.array([customer[3] for customer in all_customers])   # demand of customer t


    # Create a new model
    m = Model("assignment")

    if divisible:
        # Create variables
        z = m.addVars(T, F, vtype=GRB.CONTINUOUS, name="z")

        if divisible_except_first_one:
            # Make the first row binary
            for f in F:
                z[0, f].vtype = GRB.BINARY

        # Set objective
        m.setObjective(sum(z[t,f]*d[t,f] for t in T for f in F), GRB.MINIMIZE)

        # Add constraints
        m.addConstrs((sum(z[t,f] for f in F) == s[t] for t in T), "c1")
        m.addConstrs((sum(z[t,f] for t in T) <= c[f] for f in F), "c2")

    else:
        # Create variables
        z = m.addVars(T, F, vtype=GRB.BINARY, name="z")

        # Set objective
        m.setObjective(sum(z[t,f]*d[t,f] for t in T for f in F), GRB.MINIMIZE)

        # Add constraints
        m.addConstrs((sum(z[t,f] for f in F) == 1 for t in T), "c1")
        m.addConstrs((sum(z[t,f] for t in T) <= c[f] for f in F), "c2")

    # Make the model quiet
    m.setParam('OutputFlag', 0)
    m.setParam('Seed', GUROBI_SEED)

    '''
    # Optimize model, capped at TIME_LIMIT_SECONDS
    m.setParam('TimeLimit', TIME_LIMIT_SECONDS)
    m.optimize()

    if m.SolCount == 0:
        # no incumbent within the time limit - keep solving with no time limit, but
        # stop as soon as the first feasible solution is found, so we always return
        # an assignment instead of failing on the .x/.ObjVal reads below
        m.setParam('TimeLimit', GRB.INFINITY)
        m.setParam('SolutionLimit', 1)
        m.optimize()
    # NOTE: no m.Status check - on large instances without a full Gurobi license, this
    # can fail to reach GRB.OPTIMAL and the .x/.ObjVal/.Pi reads below will raise an
    # unhelpful AttributeError instead of a clear "no license"/"infeasible" message
    '''

    m.optimize()

    #Get the assignments
    assignments = []
    for t in T:
        for f in F:
            if z[t,f].x > 0.5:
                assignments.append(f)
                break
    
    #Get the dual values from constraint (c2)
    # opportunity cost lambda_t,f in Eq. (24): approximated as the dual value of the
    # capacity constraint (27), following the LPH heuristic (Section 5.5.2)
    if divisible and not divisible_except_first_one:
        dual_values = [m.getConstrByName('c2['+str(f)+']').getAttr('Pi') for f in F]
    else:
        dual_values = None

    return assignments, m.ObjVal, dual_values


if __name__ == "__main__":
    # smoke test: solve the offline MIP (Eqs. 1-4) on a fixed demo instance and check
    # it returns one facility per order, respecting each facility's capacity
    from miscelaneous import read_instance
    from env import InventoryEnv

    instance = read_instance('instances/instances_test/instances_seed_1.json')

    env = InventoryEnv(3, 50, 'uneven')
    env.set_instance(instance)

    assignments, obj_val, _ = perfect_hindsight(
        instance, env.warehouses_location, env.warehouses_initial_capacity, env.num_customers
    )

    assert len(assignments) == env.num_customers, "not every order got an assignment"
    for f, cap in enumerate(env.warehouses_initial_capacity):
        assert assignments.count(f) <= cap, f"facility {f} assigned beyond its capacity"
    print(f"OK: assigned {len(assignments)} orders, total distance {obj_val:.2f}")


