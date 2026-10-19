"""
It is wished to impose the following extra conditions on the food manufacture
problem:
1. The food may never be made up of more than three oils in any month.
2. If an oil is used in a month, at least 20 tons must be used.
3. If either of VEG 1 or VEG 2 are used in a month then OIL 3 must also
be used.
"""

import pyomo.environ as pyo

from lp_playground.model_building_in_mathematical_programming_problem_sets._12_1_food_manufacture_1 import (
    add_inventory_constraints,
    build_static_info,
    set_objective,
)
from lp_playground.model_building_in_mathematical_programming_problem_sets._12_1_food_manufacture_1 import (
    build_month_model as base_month_model,
)


def _add_binary_variables(block):
    model = block.model()
    block.use_oil = pyo.Var(model.oils, domain=pyo.Binary)


def _add_binary_constraints(block):
    model = block.model()

    block.oil_used_link = pyo.Constraint(
        model.oils,
        rule=lambda _block, oil: _block.oil_refined[oil]
        <= model.oil_inventory_limit * _block.use_oil[oil],
    )

    block.max_oils_in_blend = pyo.Constraint(
        name="Max 3 blends",
        expr=sum(block.use_oil[oil] for oil in model.oils) <= model.oils_in_blend_upper,
    )

    block.min_oil_tonne_used = pyo.Constraint(
        model.oils,
        name="Use 20T minimum",
        rule=lambda _block, oil: _block.oil_refined[oil]
        >= model.min_oil_use * _block.use_oil[oil],
    )

    block.oil_3_logical = pyo.Constraint(
        name="OIL 3 used with VEG 1 or VEG 2",
        expr=(block.use_oil["VEG 1"] + block.use_oil["VEG 2"])
        <= 2 * block.use_oil["OIL 3"],
    )


def build_month_model(block, month: str):
    base_month_model(block, month)
    _add_binary_variables(block)
    _add_binary_constraints(block)


def _add_binary_params(model):
    model.min_oil_use = pyo.Param(initialize=20)
    model.oils_in_blend_upper = pyo.Param(initialize=3)


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="food_manufacture_2")
    build_static_info(model)
    _add_binary_params(model)
    model.month = pyo.Block(model.months, rule=build_month_model)
    add_inventory_constraints(model)
    set_objective(model)
    results = solver.solve(model)
    print(results)
    for month in model.months:
        print(model.month[month].display())


if __name__ == "__main__":
    main()
