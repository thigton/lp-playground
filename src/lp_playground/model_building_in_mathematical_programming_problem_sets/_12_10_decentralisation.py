"""
12.10 Decentralisation
A large company wishes to move some of its departments out of London. There
are benefits to be derived from doing this (cheaper housing, government incen-
tives, easier recruitment, etc.), which have been costed. Also, however, there will
be greater costs of communication between departments. These have also been
costed for all possible locations of each department.
Where should each department be located so as to minimise overall yearly
cost?
The company comprises five departments (A, B, C, D and E). The possible
cities for relocation are Bristol and Brighton, or a department may be kept in
London. None of these cities (including London) may be the location for more
than three of the departments.
Benefits to be derived from each relocation are given (in thousands of pounds
per year) as follows:
            A   B   C   D   E
Bristol     10  15  10  20  5
Brighton    10  20  15  15  15

Communication costs are of the form C ik Djl , where C ik is the quantity of
communication between departments i and k per year and Djl is the cost per unit
of communication between cities j and l. C ik and Djl are given by the following
tables:
Quantities of communication C ik
(in thousands of units)
    A   B   C   D   E
A   –   0.0 1.0 1.5 0.0
B   –   –   1.4 1.2 0.0
C   –   –   –   0.0 2.0
D   –   –   –   –   0.7
Costs per unit of communication Djl
(in £)
            Bristol     Brighton    London
Bristol     5           14          13
Brighton    –           5           9
London      –           –           10
"""

import pyomo.environ as pyo
import itertools
from collections import defaultdict

COMM_COST = {
            ("Bristol", "Bristol"): 5,
            ("Bristol", "Brighton"): 14,
            ("Bristol", "London"): 13,
            ("Brighton", "Brighton"): 5,
            ("Brighton", "London"): 9,
            ("London", "London"): 10,
        }


def _add_sets(model):
    model.locations = pyo.Set(initialize=["Bristol", "Brighton", "London"])
    model.departments = pyo.Set(initialize=["A", "B", "C", "D", "E"])


def _add_params(model):
    model.relocation_benefit = pyo.Param(
        model.locations,
        model.departments,
        initialize={
            ("Bristol", "A"): 10,
            ("Bristol", "B"): 15,
            ("Bristol", "C"): 10,
            ("Bristol", "D"): 20,
            ("Bristol", "E"): 5,
            ("Brighton", "A"): 10,
            ("Brighton", "B"): 20,
            ("Brighton", "C"): 15,
            ("Brighton", "D"): 15,
            ("Brighton", "E"): 15,
        },
        default=0,
    )

    model.comm_qty = pyo.Param(
        model.departments,
        model.departments,
        initialize={
            ("A", "B"): 0.0,
            ("A", "C"): 1.0,
            ("A", "D"): 1.5,
            ("A", "E"): 0.0,
            ("B", "C"): 1.4,
            ("B", "D"): 1.2,
            ("B", "E"): 0.0,
            ("C", "D"): 0.0,
            ("C", "E"): 2.0,
            ("D", "E"): 0.7,
        },
    )

    model.cost_per_comm = pyo.Param(
        model.locations,
        model.locations,
        initialize=lambda _m, l1, l2: \
            COMM_COST.get((l1, l2), COMM_COST.get((l2, l1), 0)),
    )

    model.max_departments_per_city = pyo.Param(initialize=3)


def _add_variables(model):
    model.relocation = pyo.Var(
        model.locations,
        model.departments,
        domain=pyo.Binary,
    )


def _add_constraints(model):
    # department can only be in 1 city
    model.department_loc_unique = pyo.Constraint(
        model.departments,
        rule=lambda _m, dep: sum(_m.relocation[loc, dep] for loc in _m.locations)
        == 1,
    )

    model.department_per_city = pyo.Constraint(
        model.locations,
        rule=lambda _m, loc: sum(
            _m.relocation[loc, dep] for dep in _m.departments
        )
        <= _m.max_departments_per_city,
    )


def _add_objective(model):
    def _relocation_benefit_rule(_m):
        return sum(
            model.relocation[loc, dep] * model.relocation_benefit[loc, dep]
            for loc, dep in itertools.product(model.locations, model.departments)
        )

    model.total_relocation_benefit = pyo.Expression(rule=_relocation_benefit_rule)

    depts = list(model.departments)
    # each unordered department pair once
    model.dept_pairs = pyo.Set(
        dimen=2, initialize=list(itertools.combinations(depts, 2))
    )

    # y[a, b, la, lb] = 1 if dept a is in la AND dept b is in lb
    model.pair_loc = pyo.Var(
        model.dept_pairs,
        model.locations,
        model.locations,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
    )

    # Summing y over b's location recovers a's location choice
    def link_a_rule(_m, dep_a, dep_b, loc_a):
        return (
            sum(_m.pair_loc[dep_a, dep_b, loc_a, loc_b] for loc_b in _m.locations)
            == _m.relocation[loc_a, dep_a]
        )

    model.pair_link_a = pyo.Constraint(
        model.dept_pairs, model.locations, rule=link_a_rule
    )

    # Summing y over a's location recovers b's location choice
    def link_b_rule(_m, dep_a, dep_b, loc_b):
        return (
            sum(_m.pair_loc[dep_a, dep_b, loc_a, loc_b] for loc_a in _m.locations)
            == _m.relocation[loc_b, dep_b]
        )

    model.pair_link_b = pyo.Constraint(
        model.dept_pairs, model.locations, rule=link_b_rule
    )

    def comm_cost_rule(_m):
        return sum(
            _m.comm_qty[a, b] * _m.cost_per_comm[la, lb] * _m.pair_loc[a, b, la, lb]
            for a, b in _m.dept_pairs
            for la, lb in itertools.product(_m.locations, repeat=2)
        )

    model.comm_cost = pyo.Expression(rule=comm_cost_rule)

    model.OBJ = pyo.Objective(
        expr=model.total_relocation_benefit - model.comm_cost,
        sense=pyo.maximize)


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="decentralisation")
    _add_sets(model)
    _add_params(model)
    _add_variables(model)
    _add_constraints(model)
    _add_objective(model)
    results = solver.solve(model)
    print(results)
    locations = defaultdict(list)
    for loc, dep in itertools.product(model.locations, model.departments):
        if pyo.value(model.relocation[loc, dep]) > 0.5:
            locations[loc].append(dep)
    print(locations)




if __name__ == "__main__":
    main()
