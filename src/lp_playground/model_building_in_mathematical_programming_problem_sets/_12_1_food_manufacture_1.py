"""
12.1 Food manufacture 1
A food is manufactured by refining raw oils and blending them together. The
raw oils are of two categories:
Vegetable oils
VEG 1
VEG 2
Non-vegetable oils
OIL 1
OIL 2
OIL 3

Each oil may be purchased for immediate delivery (January) or bought on the
futures market for delivery in a subsequent month. Prices at present and in the
futures market are given below in (£/ton):
            VEG 1   VEG 2   OIL 1   OIL 2   OIL 3
January     110     120     130     110     115
February    130     130     110     90      115
March       110     140     130     100     95
April       120     110     120     120     125
May         100     120     150     110     105
June        90      100     140     80      135

The final product sells at £150 per ton.
Vegetable oils and non-vegetable oils require different production lines for
refining. In any month, it is not possible to refine more than 200 tons of vegetable
oils and more than 250 tons of non-vegetable oils. There is no loss of weight in
the refining process, and the cost of refining may be ignored.
It is possible to store up to 1000 tons of each raw oil for use later. The cost
of storage for vegetable and non-vegetable oil is £5 per ton per month. The final
product cannot be stored, nor can refined oils be stored.
There is a technological restriction of hardness on the final product. In the
units in which hardness is measured, this must lie between 3 and 6. It is assumed
that hardness blends linearly and that the hardnesses of the raw oils are
VEG 1 8.8
VEG 2 6.1
OIL 1 2.0
OIL 2 4.2
OIL 3 5.0
What buying and manufacturing policy should the company pursue in order to
maximise profit?
At present, there are 500 tons of each type of raw oil in storage. It is required
that these stocks will also exist at the end of June.

"""

import itertools

import pandas as pd
import pyomo.environ as pyo

OILS = {
    "vegetable oils": ["VEG 1", "VEG 2"],
    "non-vegetable oils": ["OIL 1", "OIL 2", "OIL 3"],
}
OILS_LIST = list(itertools.chain.from_iterable(OILS.values()))
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]


def build_costs():
    costs = pd.DataFrame(
        data=[
            [110, 120, 130, 110, 115],  # Jan
            [130, 130, 110, 90, 115],  # Feb
            [110, 140, 130, 100, 95],
            [120, 110, 120, 120, 125],
            [100, 120, 150, 110, 105],
            [90, 100, 140, 80, 135],
        ],
        columns=OILS_LIST,
        index=MONTHS,
    )
    return costs.stack().to_dict()


def build_static_info(model):
    model.months = pyo.Set(initialize=MONTHS, ordered=True)
    _add_oil_info(model)
    _add_product_info(model)
    model.storage_cost = pyo.Param(initialize=5)
    model.oil_inventory_limit = pyo.Param(initialize=1000)


def _add_oil_info(model):
    model.oils = pyo.Set(initialize=OILS_LIST)
    model.oil_types = pyo.Set(initialize=list(OILS.keys()))
    model.oils_by_type = pyo.Set(model.oil_types, initialize=OILS)
    model.oil_type_refinement_limit = pyo.Param(
        model.oil_types,
        initialize={
            "vegetable oils": 200,
            "non-vegetable oils": 250,
        },
    )
    model.oil_hardness = pyo.Param(
        model.oils,
        initialize={
            "VEG 1": 8.8,
            "VEG 2": 6.1,
            "OIL 1": 2.0,
            "OIL 2": 4.2,
            "OIL 3": 5.0,
        },
    )
    costs = {k: v for k, v in build_costs().items() if k[0] in model.months}
    model.costs = pyo.Param(model.months, model.oils, initialize=costs)


def _add_product_info(model):
    model.prod_price = pyo.Param(initialize=150)
    model.prod_hardness_lower = pyo.Param(initialize=3)
    model.prod_hardness_upper = pyo.Param(initialize=6)


def _add_variables(month_block):
    model = month_block.model()
    month_block.oil_purchase = pyo.Var(model.oils, domain=pyo.NonNegativeReals)
    month_block.oil_inventory = pyo.Var(
        model.oils, domain=pyo.NonNegativeReals, bounds=(0, model.oil_inventory_limit)
    )
    month_block.oil_refined = pyo.Var(model.oils, domain=pyo.NonNegativeReals)
    month_block.prod = pyo.Var(domain=pyo.NonNegativeReals)


def _add_expressions(month_block):
    model = month_block.model()
    month_block.total_hardness = pyo.Expression(
        expr=sum(
            month_block.oil_refined[oil] * model.oil_hardness[oil] for oil in model.oils
        )
    )

    month_block.revenue = pyo.Expression(expr=model.prod_price * month_block.prod)

    month_block.purchase_cost = pyo.Expression(
        expr=sum(
            model.costs[month_block.index(), oil] * month_block.oil_purchase[oil]
            for oil in model.oils
        )
    )

    month_block.storage_cost = pyo.Expression(
        expr=sum(
            model.storage_cost * month_block.oil_inventory[oil] for oil in model.oils
        )
    )

    month_block.profit = pyo.Expression(
        expr=(
            month_block.revenue - month_block.purchase_cost - month_block.storage_cost
        )
    )


def _add_constraints(month_block):
    model = month_block.model()

    def _refinement_rule(_block, oil_type):
        return (
            sum(_block.oil_refined[oil] for oil in model.oils_by_type[oil_type])
            <= model.oil_type_refinement_limit[oil_type]
        )

    month_block.refinement_limit = pyo.Constraint(
        model.oil_types,
        rule=_refinement_rule,
    )

    def _lower_hardness_rule(_block):
        return (
            sum(_block.oil_refined[oil] * model.oil_hardness[oil] for oil in model.oils)
            >= model.prod_hardness_lower * _block.prod
        )

    month_block.hardness_lower = pyo.Constraint(
        rule=_lower_hardness_rule,
    )

    def _upper_hardness_rule(_block):
        return (
            sum(_block.oil_refined[oil] * model.oil_hardness[oil] for oil in model.oils)
            <= model.prod_hardness_upper * _block.prod
        )

    month_block.hardness_upper = pyo.Constraint(
        rule=_upper_hardness_rule,
    )

    def __volume_rule(_block):
        return sum(_block.oil_refined[oil] for oil in model.oils) == _block.prod

    month_block.volume_in_month = pyo.Constraint(rule=__volume_rule)


def add_inventory_constraints(model):
    model.final_inventory = pyo.Constraint(
        model.oils,
        rule=lambda _model, oil: _model.month[_model.months.last()].oil_inventory[oil]
        == 500,
    )

    def _inventory_balance_rule(_model, month, oil):
        if month == _model.months.first():
            prev_month_inventory = 500
        else:
            prev_month_inventory = _model.month[
                _model.months.prev(month)
            ].oil_inventory[oil]
        month_b = _model.month[month]
        return (
            month_b.oil_inventory[oil]
            == prev_month_inventory
            + month_b.oil_purchase[oil]
            - month_b.oil_refined[oil]
        )

    model.inventory_balance = pyo.Constraint(
        model.months, model.oils, rule=_inventory_balance_rule
    )


def build_month_model(block, month: str):
    _add_variables(block)
    _add_constraints(block)
    _add_expressions(block)


def _check_oil_inventory_balance(model, oil: str):
    print("=" * 20)
    print(oil)
    print("-" * 20)
    for month in model.months:
        block = model.month[month]

        if month == model.months.first():
            print(
                month,
                "Initial:",
                500,
                "Purchase:",
                round(block.oil_purchase[oil].value, 2),
                "Refined:",
                round(block.oil_refined[oil].value, 2),
                "Inventory:",
                round(block.oil_inventory[oil].value, 2),
            )
        else:
            previous_month = model.months.prev(month)
            previous_inventory = model.month[previous_month].oil_inventory[oil].value

            calculated_inventory = (
                previous_inventory
                + block.oil_purchase[oil].value
                - block.oil_refined[oil].value
            )

            print(
                month,
                "Previous:",
                round(previous_inventory, 2),
                "+ Purchase:",
                round(block.oil_purchase[oil].value, 2),
                "- Refined:",
                round(block.oil_refined[oil].value, 2),
                "= Calculated:",
                round(calculated_inventory, 2),
                "| Actual:",
                round(block.oil_inventory[oil].value, 2),
            )


def set_objective(model):
    model.objective = pyo.Objective(
        expr=sum(model.month[month].profit for month in model.months),
        sense=pyo.maximize,
    )


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="food_manufacture_1")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    build_static_info(model)
    model.month = pyo.Block(model.months, rule=build_month_model)
    add_inventory_constraints(model)
    set_objective(model)
    results = solver.solve(model)
    print(results)
    # print(model.display())
    hardness = {}
    for month in model.months:
        print(model.month[month].display())

        hardness[month] = (
            model.month[month].total_hardness() / model.month[month].prod.value
        )
    print(hardness)
    for oil in model.oils:
        _check_oil_inventory_balance(model, oil)
    # display all duals
    print("Duals")
    for c in model.component_objects(pyo.Constraint, active=True):
        print("   Constraint", c)
        for index in c:
            print("      ", index, model.dual[c[index]])


if __name__ == "__main__":
    main()
