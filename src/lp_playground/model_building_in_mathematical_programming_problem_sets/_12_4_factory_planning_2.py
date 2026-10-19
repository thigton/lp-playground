import pyomo.environ as pyo

from lp_playground.model_building_in_mathematical_programming_problem_sets._12_3_factory_planning_1 import (
    MONTHS,
    PROCESS,
    add_expressions,
    add_inventory_constraints,
    add_product_constants,
    add_production_constants,
    add_storage_constants,
    add_variables,
    set_objective,
)


def _add_machine_constants(model) -> None:
    model.machines = pyo.Set(initialize=PROCESS)
    model.n_machines = pyo.Param(
        model.machines,
        initialize=dict(zip(PROCESS, [4, 2, 3, 1, 1])),
        units=pyo.units.machine,
    )
    model.n_machines_to_maintain = pyo.Param(
        model.machines,
        initialize=dict(zip(PROCESS, [2, 2, 3, 1, 1])),
        units=pyo.units.machine,
    )


def add_constants(model):
    model.months = pyo.Set(initialize=MONTHS, ordered=True)
    _add_machine_constants(model)
    add_product_constants(model)
    add_storage_constants(model)
    add_production_constants(model)


def add_constraints(month_block: pyo.Block, month: str):
    model = month_block.model()

    def _machine_capacity_rule(_block, machine: str):
        # machine production time * product_manufactured for all products <
        # capacity

        return sum(
            model.production_times[machine, product]
            * _block.product_manufactured[product]
            for product in model.products
        ) <= model.working_hours_per_month * (
            model.n_machines[machine] - _block.machines_in_maintenance[machine]
        )

    month_block.machine_capacity = pyo.Constraint(
        model.machines, rule=_machine_capacity_rule
    )

    month_block.market_demand_limit = pyo.Constraint(
        model.products,
        expr=lambda _block, product: _block.product_sold[product]
        <= model.market_demand[month, product],
    )


def _add_binary_variables(month_block):
    model = month_block.model()
    month_block.machines_in_maintenance = pyo.Var(
        model.machines,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda _block, machine: (0, model.n_machines_to_maintain[machine]),
    )


def build_month_model(block: pyo.Block, month: str):
    add_variables(month_block=block)
    _add_binary_variables(month_block=block)
    add_constraints(month_block=block, month=month)
    add_expressions(month_block=block)


def _add_total_maintenance_constraint(model):
    def _machines_to_maintain_limit(_model, machine):
        return (
            sum(
                _model.month[month].machines_in_maintenance[machine]
                for month in _model.months
            )
            == _model.n_machines_to_maintain[machine]
        )

    model.total_machine_maintenance = pyo.Constraint(
        model.machines, rule=_machines_to_maintain_limit
    )


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="factory_planning_2")
    add_constants(model)
    model.month = pyo.Block(model.months, rule=build_month_model)
    _add_total_maintenance_constraint(model)
    add_inventory_constraints(model)
    set_objective(model)
    results = solver.solve(model)
    print(results)

    for month in model.months:
        print(model.month[month].display())


if __name__ == "__main__":
    main()
