"""
A company is undergoing a number of changes that will affect its manpower
requirements in future years. Owing to the installation of new machinery, fewer
unskilled but more skilled and semi-skilled workers will be required. In addition
to this, a downturn in trade is expected in the next year, which will reduce the
need for workers in all categories. The estimated manpower requirements for the
next three years are as follows:
                    Unskilled   Semi-skilled    Skilled
Current strength    2000        1500            1000
Year 1              1000        1400            1000
Year 2              500         2000           1500
Year 3              0           2500            2000

The company wishes to decide its policy with regard to the following over
the next three years:
1. Recruitment
2. Retraining
3. Redundancy
4. Short-time working.
There is a natural wastage of labour. A fairly large number of workers leave
during their first year. After this, the rate is much smaller. Taking this into
account, the wastage rates can be taken as follows:
                                Unskilled (%)   Semi-skilled (%)    Skilled (%)
Less than one year’s service    25              20                  10
More than one year’s service    10              5                   5
There has been no recent recruitment and all workers in the current labour
force have been employed for more than one year.

12.5.1 Recruitment
It is possible to recruit a limited number of workers from outside. In any one
year, the numbers that can be recruited in each category are as follows:
Unskilled   Semi-skilled    Skilled
500         800             500

12.5.2 Retraining
It is possible to retrain up to 200 unskilled workers per year to make them semi-
skilled. This costs £400 per worker. The retraining of semi-skilled workers to
make them skilled is limited to no more than one quarter of the skilled labour
force at the time as some training is done on the job. Retraining a semi-skilled
worker in this way costs £500.
Downgrading of workers to a lower skill is possible but 50% of such workers
leave, although it costs the company nothing. (This wastage is additional to the
‘natural wastage’ described above).

12.5.3 Redundancy
The redundancy payment to an unskilled worker is £200 and to a semi-skilled
or skilled worker is £500.

12.5.4 Overmanning
It is possible to employ up to 150 more workers over the whole company than
are needed, but the extra costs per employee per year are as follows:
Unskilled   Semi-skilled    Skilled
£1500       £2000           £3000

12.5.5 Short-time working
Up to 50 workers in each category of skill can be put on short-time working.
The cost of this (per employee per year) is as follows:
Unskilled   Semi-skilled    Skilled
£500        £400            £400
An employee on short-time working meets the production requirements of
half a full-time employee.

The company’s declared objective is to minimise redundancy. How should
they operate in order to do this?
If their policy were to minimise costs, how much extra would this save?
Deduce the cost of saving each type of job each year.
"""

import itertools

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

pyo.units.load_definitions_from_strings(
    [
        "GBP = [currency]",
        "worker = [worker]",
    ]
)

SKILLS = ["Unskilled", "Semi-skilled", "Skilled"]
YEARS_SERVICE = ["<1", ">=1"]

YEARS = ["0", "1", "2", "3"]

MANPOWER_REQUIREMENTS = pd.DataFrame(
    data=[
        [2000, 1500, 1000],
        [1000, 1400, 1000],
        [500, 2000, 1500],
        [0, 2500, 2000],
    ],
    index=YEARS,
    columns=SKILLS,
)

LABOUR_WASTAGE = pd.DataFrame(
    data=[
        [0.25, 0.2, 0.1],
        [0.1, 0.05, 0.05],
    ],
    index=YEARS_SERVICE,
    columns=SKILLS,
)


SKILLS = ["Unskilled", "Semi-skilled", "Skilled"]


def add_indexes(model) -> None:
    model.skills = pyo.Set(initialize=SKILLS, ordered=True)
    model.retrain_pairs = pyo.Set(
        dimen=2,
        initialize=[
            ("Unskilled", "Semi-skilled"),
            ("Semi-skilled", "Skilled"),
        ],
    )
    model.downgrade_pairs = pyo.Set(
        dimen=2,
        initialize=[
            ("Skilled", "Semi-skilled"),
            ("Skilled", "Unskilled"),
            ("Semi-skilled", "Unskilled"),
        ],
    )

    model.years_service = pyo.Set(initialize=YEARS_SERVICE)
    model.years = pyo.Set(initialize=YEARS, ordered=True)


def add_params(model) -> None:
    # manpower params
    model.manpower_requirements = pyo.Param(
        model.years,
        model.skills,
        initialize=MANPOWER_REQUIREMENTS.stack().to_dict(),
        units=pyo.units.worker,
    )

    # wastage params
    model.labour_wastage = pyo.Param(
        model.years_service,
        model.skills,
        initialize=LABOUR_WASTAGE.stack().to_dict(),
    )
    model.downgrading_wastage = pyo.Param(
        initialize=0.5,
    )

    # recruitment params
    model.recruitment_limit = pyo.Param(
        model.skills,
        initialize=dict(zip(SKILLS, [500, 800, 500])),
        units=pyo.units.worker,
    )

    # retraining params
    model.unskilled_retraining_abs_limit = pyo.Param(
        initialize=200,
        units=pyo.units.worker,
    )
    model.semi_skilled_retraining_frac_limit = pyo.Param(initialize=0.25)

    model.retraining_cost_per_worker = pyo.Param(
        model.retrain_pairs,
        initialize={
            ("Unskilled", "Semi-skilled"): 400,
            ("Semi-skilled", "Skilled"): 500,
        },
        units=pyo.units.GBP / pyo.units.worker,
    )

    # redundancy params
    model.redundancy_cost_per_worker = pyo.Param(
        model.skills,
        initialize=dict(zip(SKILLS, [200, 500, 500])),
        units=pyo.units.GBP,
    )

    # overmanning params
    model.overmanning_limit = pyo.Param(
        initialize=150,
        units=pyo.units.worker,
    )

    model.overmanning_cost_per_worker = pyo.Param(
        model.skills,
        initialize=dict(zip(SKILLS, [1500, 2000, 3000])),
        units=pyo.units.GBP,
    )

    # short time worker params
    model.short_time_worker_limit = pyo.Param(
        initialize=50,
        units=pyo.units.worker,
    )

    model.short_time_cost_per_worker = pyo.Param(
        model.skills,
        initialize=dict(zip(SKILLS, [500, 400, 400])),
        units=pyo.units.GBP,
    )
    model.short_time_worker_production_frac = pyo.Param(initialize=0.5)


def _add_variables(block):
    model = block.model()
    block.workforce = pyo.Var(
        model.skills,
        domain=pyo.NonNegativeReals,
    )
    block.workers_recruited = pyo.Var(
        model.skills,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda _block, skill: (0, model.recruitment_limit[skill]),
    )

    block.workers_retrained = pyo.Var(
        model.retrain_pairs,
        domain=pyo.NonNegativeIntegers,
    )

    block.workers_downgraded = pyo.Var(
        model.downgrade_pairs,
        domain=pyo.NonNegativeIntegers,
    )

    block.workers_redundant = pyo.Var(
        model.skills,
        domain=pyo.NonNegativeIntegers,
    )
    block.workers_short_time = pyo.Var(
        model.skills,
        domain=pyo.NonNegativeIntegers,
        bounds=(0, model.short_time_worker_limit),
    )

    block.workers_overmanned = pyo.Var(
        model.skills,
        domain=pyo.NonNegativeIntegers,
        bounds=(0, model.overmanning_limit),
    )


def _manpower_constraints(block, year: str) -> None:
    model = block.model()
    # Fix workforce at each year to the requirements

    def _manpower_rule(_block, skill):
        return (
            _block.workforce[skill]
            - _block.workers_overmanned[skill]
            - model.short_time_worker_production_frac * _block.workers_short_time[skill]
            == model.manpower_requirements[year, skill]
        )

    block.manpower_req = pyo.Constraint(
        model.skills,
        rule=_manpower_rule,
    )

    def _manpower_continuity_rule(_block, skill):
        if year == model.years.first():
            return pyo.Constraint.Skip
        curr_workforce = _block.workforce[skill]
        prev_workforce = model.year[model.years.prev(year)].workforce[skill] * (
            1 - model.labour_wastage[">=1", skill]
        )
        recruitment = _block.workers_recruited[skill] * (
            1 - model.labour_wastage["<1", skill]
        )
        redundancy = _block.workers_redundant[skill]

        retrain_in = (1 - model.labour_wastage[">=1", skill]) * sum(
            _block.workers_retrained[from_skill, to_skill]
            for from_skill, to_skill in model.retrain_pairs
            if to_skill == skill
        )

        retrain_out = sum(
            _block.workers_retrained[from_skill, to_skill]
            for from_skill, to_skill in model.retrain_pairs
            if from_skill == skill
        )

        downgrade_in = (1 - model.downgrading_wastage) * sum(
            _block.workers_downgraded[from_skill, to_skill]
            for from_skill, to_skill in model.downgrade_pairs
            if to_skill == skill
        )

        downgrade_out = sum(
            _block.workers_downgraded[from_skill, to_skill]
            for from_skill, to_skill in model.downgrade_pairs
            if from_skill == skill
        )
        return (
            prev_workforce
            + recruitment
            - redundancy
            + retrain_in
            - retrain_out
            + downgrade_in
            - downgrade_out
            == curr_workforce
        )

    block.manpower_continuity = pyo.Constraint(
        model.skills, rule=_manpower_continuity_rule
    )


def _retraining_constraints(block) -> None:
    model = block.model()
    block.retrain_unskilled_upper_limit = pyo.Constraint(
        expr=block.workers_retrained["Unskilled", "Semi-skilled"]
        <= model.unskilled_retraining_abs_limit
    )
    block.retrain_semi_skilled_upper = pyo.Constraint(
        expr=(
            block.workers_retrained["Semi-skilled", "Skilled"]
            <= model.semi_skilled_retraining_frac_limit * block.workforce["Skilled"]
        )
    )


def _overmanning_constraints(block) -> None:
    model = block.model()
    block.overmanning_upper = pyo.Constraint(
        expr=sum(block.workers_overmanned[skill] for skill in model.skills)
        <= model.overmanning_limit
    )


def _cost_expressions(block):
    model = block.model()
    block.retraining_costs = pyo.Expression(
        expr=sum(
            block.workers_retrained[from_skill, to_skill]
            * model.retraining_cost_per_worker[from_skill, to_skill]
            for from_skill, to_skill in model.retrain_pairs
        )
    )

    block.redundancy_cost = pyo.Expression(
        expr=sum(
            block.workers_redundant[skill] * model.redundancy_cost_per_worker[skill]
            for skill in model.skills
        )
    )

    block.overmanning_cost = pyo.Expression(
        expr=sum(
            block.workers_overmanned[skill] * model.overmanning_cost_per_worker[skill]
            for skill in model.skills
        )
    )

    block.short_time_cost = pyo.Expression(
        expr=sum(
            block.workers_short_time[skill] * model.short_time_cost_per_worker[skill]
            for skill in model.skills
        )
    )


def build_year_block(block, year: str) -> None:
    _add_variables(block)
    _manpower_constraints(block, year)
    _retraining_constraints(block)
    _overmanning_constraints(block)
    _cost_expressions(block)


def _minimize_redundancy(model) -> None:
    model.total_redundancy = pyo.Expression(
        expr=sum(
            model.year[year].workers_redundant[skill]
            for year, skill in itertools.product(model.years, model.skills)
        )
    )


def _minimize_cost(model) -> None:
    model.total_retraining_cost = pyo.Expression(
        expr=sum(model.year[year].retraining_costs for year in model.years)
    )

    model.total_redundancy_cost = pyo.Expression(
        expr=sum(model.year[year].redundancy_cost for year in model.years)
    )

    model.total_overmanning_cost = pyo.Expression(
        expr=sum(model.year[year].overmanning_cost for year in model.years)
    )

    model.total_short_time_cost = pyo.Expression(
        expr=sum(model.year[year].short_time_cost for year in model.years)
    )

    model.total_cost = (
        model.total_retraining_cost
        + model.total_redundancy_cost
        + model.total_overmanning_cost
        + model.total_short_time_cost
    )


def set_objectives(model):
    _minimize_redundancy(model)
    model.redundancy_objective = pyo.Objective(
        expr=model.total_redundancy,
        sense=pyo.minimize,
    )
    _minimize_cost(model)
    model.cost_objective = pyo.Objective(
        expr=model.total_cost,
        sense=pyo.minimize,
    )

    model.redundancy_objective.deactivate()
    model.cost_objective.deactivate()


def set_weighted_objective(model, coeffs):
    model.objective_weight = pyo.Param(mutable=True, initialize=0)

    scaled_redundancy = (model.total_redundancy - coeffs["r_min"]) / (
        coeffs["r_max"] - coeffs["r_min"]
    )
    scaled_cost = (model.total_cost - coeffs["c_min"]) / (
        coeffs["c_max"] - coeffs["c_min"]
    )
    model.weighted_objective = pyo.Objective(
        expr=(
            model.objective_weight * scaled_redundancy
            + (1 - model.objective_weight) * scaled_cost
        ),
        sense=pyo.minimize,
    )


def _get_pareto_front_coefficients(model, solver):
    model.redundancy_objective.activate()
    solver.solve(model)
    redundancy_min, cost_max = model.total_redundancy(), model.total_cost()
    model.redundancy_objective.deactivate()
    model.cost_objective.activate()
    solver.solve(model)
    redundancy_max, cost_min = model.total_redundancy(), model.total_cost()
    model.cost_objective.deactivate()
    return {
        "r_min": redundancy_min,
        "r_max": redundancy_max,
        "c_min": cost_min,
        "c_max": cost_max,
    }


def _fix_initial_manpower_conditions(model) -> None:
    for skill in model.skills:
        model.year["0"].workforce[skill].fix(model.manpower_requirements["0", skill])
    model.year["0"].workers_short_time[:].fix(0)
    model.year["0"].workers_overmanned[:].fix(0)
    model.year["0"].workers_recruited[:].fix(0)
    model.year["0"].workers_retrained[:, :].fix(0)
    model.year["0"].workers_downgraded[:, :].fix(0)
    model.year["0"].workers_redundant[:].fix(0)


def print_result(results):
    if (
        results.solver.status == pyo.SolverStatus.ok
        and results.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        print("Optimal solution found")
        print(results)
    else:
        print("Something went wrong:")
        print(results.solver.status)
        print(results.solver.termination_condition)


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="manpower_planning")
    add_indexes(model)
    add_params(model)
    model.year = pyo.Block(model.years, rule=build_year_block)
    _fix_initial_manpower_conditions(model)
    set_objectives(model)
    pareto_coeffs = _get_pareto_front_coefficients(model, solver)
    set_weighted_objective(model, pareto_coeffs)
    total_redundancy = []
    total_cost = []
    for obj_weight in range(101):
        weight = obj_weight / 100
        model.objective_weight.set_value(weight)
        print(f"solving with {weight=}")
        solver.solve(model)
        redundancy = model.total_redundancy()
        cost = model.total_cost()
        print(f"Solved. {weight=} | {redundancy=} | {cost=}")
        total_cost.append(cost)
        total_redundancy.append(redundancy)

    fig, ax = plt.subplots()
    ax.plot(total_cost, total_redundancy, marker="o", markersize=3)
    ax.set_xlabel("Total cost")
    ax.set_ylabel("Total redundancy")
    ax.set_title("Cost–redundancy trade-off")

    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
