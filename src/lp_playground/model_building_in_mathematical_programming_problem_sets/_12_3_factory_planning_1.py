"""
An engineering factory makes seven products (PROD 1 to PROD 7) on the
following machines: four grinders, two vertical drills, three horizontal drills, one
borer and one planer. Each product yields a certain contribution to profit (defined
as £/unit selling price minus cost of raw materials). These quantities (in £/unit)
together with the unit production times (hours) required on each process are given
below. A dash indicates that a product does not require a process.
                        PROD 1  PROD 2  PROD 3  PROD 4  PROD 5  PROD 6  PROD 7
Contribution to profit  10      6       8       4       11      9       3
Grinding                0.5     0.7     –       –       0.3     0.2     0.5
Vertical drilling       0.1     0.2     –       0.3     –       0.6     –
Horizontal drilling     0.2     –       0.8     –       –       –       0.6
Boring                  0.05    0.03    –       0.07    0.1     –       0.08
Planing                 –       –       0.01    –       0.05    –       0.05
In the present month (January) and the five subsequent months, certain
machines will be down for maintenance. These machines will be as follows:
January 1 Grinder
February 2 Horizontal drills
March 1 Borer
April 1 Vertical drill
May 1 Grinder and 1 Vertical drill
June 1 Planer and 1 Horizontal drill

There are marketing limitations on each product in each month. These are
given in the following table:
            1   2       3   4   5   6   7
January     500 1000    300 300 800 200 100
February    600 500     200 0   400 300 150
March       300 600     0   0   500 400 100
April       200 300     400 500 200 0   100
May         0   100     500 100 1000 300 0
June        500 500     100 300 1100 500 60
It is possible to store up to 100 of each product at a time at a cost of £0.5
per unit per month. There are no stocks at present, but it is desired to have a
stock of 50 of each type of product at the end of June.
The factory works a six days a week with two shifts of 8 h each day.
No sequencing problems need to be considered.
When and what should the factory make in order to maximise the total profit?
Recommend any price increases and the value of acquiring any new machines.
N.B. It may be assumed that each month consists of only 24 working days.
"""

import pandas as pd
import pyomo.environ as pyo
from typing import Protocol

PRODUCTS = ["PROD1", "PROD2", "PROD3", "PROD4", "PROD5", "PROD6", "PROD7"]
PROCESS = ["Grinding", "Vertical Drilling", "Horizontal Drilling", "Boring", "Planing"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]

PRODUCTION_TIMES = pd.DataFrame(
    data=[
        [0.5, 0.7, 0, 0, 0.3, 0.2, 0.5],  # Grinding
        [0.1, 0.2, 0, 0.3, 0, 0.6, 0],  # Vertical drilling etc.
        [0.2, 0, 0.8, 0, 0, 0, 0.6],
        [0.05, 0.03, 0, 0.07, 0.1, 0, 0.08],
        [0, 0, 0.01, 0, 0.05, 0, 0.05],
    ],
    columns=PRODUCTS,
    index=PROCESS,
)

MARKETING_DEMANDS = pd.DataFrame(
    data=[
        [500, 1000, 300, 300, 800, 200, 100],
        [600, 500, 200, 0, 400, 300, 150],
        [300, 600, 0, 0, 500, 400, 100],
        [200, 300, 400, 500, 200, 0, 100],
        [0, 100, 500, 100, 1000, 300, 0],
        [500, 500, 100, 300, 1100, 500, 60],
    ],
    columns=PRODUCTS,
    index=MONTHS,
)

MAINTENANCE_SCHEDULE: dict[tuple[str, str], int] = {
    ("JAN", "Grinding"): 1,
    ("FEB", "Horizontal Drilling"): 2,
    ("MAR", "Boring"): 1,
    ("APR", "Vertical Drilling"): 1,
    ("MAY", "Grinding"): 1,
    ("MAY", "Vertical Drilling"): 1,
    ("JUN", "Planing"): 1,
    ("JUN", "Horizontal Drilling"): 1,
}

pyo.units.load_definitions_from_strings(
    [
        "GBP = [currency]",
        "product = [product]",
        "machine = [machine]",
    ]
)


class ModelWMonths(Protocol):
    months: pyo.Set


def _add_machine_constants(model: ModelWMonths) -> None:
    model.machines = pyo.Set(initialize=PROCESS)
    model.n_machines = pyo.Param(
        model.machines,
        initialize=dict(zip(PROCESS, [4, 2, 3, 1, 1])),
        units=pyo.units.machine,
    )
    model.machine_maintenance = pyo.Param(
        model.months,
        model.machines,
        initialize=MAINTENANCE_SCHEDULE,
        default=0,
    )


def add_product_constants(model: ModelWMonths) -> None:
    model.products = pyo.Set(initialize=PRODUCTS)
    model.product_profit = pyo.Param(
        model.products,
        initialize=dict(zip(PRODUCTS, [10, 6, 8, 4, 11, 9, 3])),
        units=pyo.units.GBP,
    )
    model.market_demand = pyo.Param(
        model.months,
        model.products,
        initialize=MARKETING_DEMANDS.stack().to_dict(),
        units=pyo.units.product,
    )


def add_storage_constants(model: pyo.Model) -> None:
    model.storage_costs_per_unit = pyo.Param(
        initialize=0.5, units=pyo.units.GBP / pyo.units.product / pyo.units.month
    )
    model.storage_limit = pyo.Param(
        initialize=100,
        units=pyo.units.product,
    )
    model.final_inventory_per_product = pyo.Param(
        initialize=50, units=pyo.units.product
    )


class ModelWSets(Protocol):
    months: pyo.Set
    products: pyo.Set
    machines: pyo.Set


def add_production_constants(model: ModelWSets) -> None:
    model.working_hours_per_day = pyo.Param(
        initialize=16, units=pyo.units.hour / pyo.units.day
    )
    model.working_days_per_month = pyo.Param(
        initialize=24, units=pyo.units.day / pyo.units.month
    )
    model.working_hours_per_month = pyo.Param(
        initialize=model.working_hours_per_day * model.working_days_per_month,
        units=pyo.units.hour / pyo.units.month,
    )
    model.production_times = pyo.Param(
        model.machines,
        model.products,
        initialize=PRODUCTION_TIMES.stack().to_dict(),
        units=pyo.units.hour / pyo.units.product,
    )


def add_constants(model) -> None:
    model.months = pyo.Set(initialize=MONTHS, ordered=True)
    _add_machine_constants(model)
    add_product_constants(model)
    add_storage_constants(model)
    add_production_constants(model)


def add_variables(month_block: pyo.Block):
    model = month_block.model()

    month_block.product_manufactured = pyo.Var(
        model.products, domain=pyo.NonNegativeReals, units=pyo.units.product
    )

    month_block.product_inventory = pyo.Var(
        model.products,
        domain=pyo.NonNegativeReals,
        bounds=(0, model.storage_limit),
        units=pyo.units.product,
    )
    month_block.product_sold = pyo.Var(
        model.products,
        domain=pyo.NonNegativeReals,
        units=pyo.units.product,
    )


class ModelForConstraints(ModelWSets):
    production_times: pyo.Param
    working_hours_per_month: pyo.Param
    n_machines: pyo.Param
    machine_maintenance: pyo.Param
    market_demand: pyo.Param


def add_constraints(month_block: pyo.Block, month: str):
    model: ModelForConstraints = month_block.model()

    def _machine_capacity_rule(_block, machine: str):
        # machine production time * product_manufactured for all products <
        # capacity

        return sum(
            model.production_times[machine, product]
            * _block.product_manufactured[product]
            for product in model.products
        ) <= model.working_hours_per_month * (
            model.n_machines[machine] - model.machine_maintenance[month, machine]
        )

    month_block.machine_capacity = pyo.Constraint(
        model.machines, rule=_machine_capacity_rule
    )

    month_block.market_demand_limit = pyo.Constraint(
        model.products,
        expr=lambda _block, product: _block.product_sold[product]
        <= model.market_demand[month, product],
    )


def add_expressions(month_block: pyo.Block) -> None:
    model = month_block.model()
    month_block.sales_profit_by_product = pyo.Expression(
        model.products,
        expr=lambda _block, product: model.product_profit[product]
        * month_block.product_sold[product],
    )
    month_block.sales_profit = pyo.Expression(
        expr=sum(
            month_block.sales_profit_by_product[product] for product in model.products
        )
    )
    month_block.storage_costs = pyo.Expression(
        expr=sum(
            model.storage_costs_per_unit * month_block.product_inventory[product]
            for product in model.products
        )
    )
    month_block.profit = month_block.sales_profit - month_block.storage_costs


def build_month_model(block: pyo.Block, month: str):
    add_variables(month_block=block)
    add_constraints(month_block=block, month=month)
    add_expressions(month_block=block)


def add_inventory_constraints(model):
    model.final_inventory = pyo.Constraint(
        model.products,
        rule=lambda _model, product: _model.month[
            _model.months.last()
        ].product_inventory[product]
        == model.final_inventory_per_product,
    )

    def _inventory_balance_rule(_model, month, product):
        if month == _model.months.first():
            prev_month_inventory = 0
        else:
            prev_month_inventory = _model.month[
                _model.months.prev(month)
            ].product_inventory[product]
        month_b = _model.month[month]
        return (
            month_b.product_inventory[product]
            == prev_month_inventory
            + month_b.product_manufactured[product]
            - month_b.product_sold[product]
        )

    model.inventory_balance = pyo.Constraint(
        model.months, model.products, rule=_inventory_balance_rule
    )


def set_objective(model):
    model.objective = pyo.Objective(
        expr=sum(model.month[month].profit for month in model.months),
        sense=pyo.maximize,
    )


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="factory_planning_1")
    add_constants(model)
    model.month = pyo.Block(model.months, rule=build_month_model)
    add_inventory_constraints(model)
    set_objective(model)
    results = solver.solve(model)
    print(results)

    for month in model.months:
        print(model.month[month].display())


if __name__ == "__main__":
    main()
