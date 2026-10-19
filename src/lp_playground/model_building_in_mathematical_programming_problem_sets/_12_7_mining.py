"""
12.7 Mining
A mining company is going to continue operating in a certain area for the next
five years. There are four mines in this area, but it can operate at most three in any
one year. Although a mine may not operate in a certain year, it is still necessary to
keep it ‘open’, in the sense that royalties are payable, if it be operated in a future
year. Clearly, if a mine is not going to be worked again, it can be permanently
closed down and no more royalties need be paid. The yearly royalties payable
on each mine kept ‘open’ are as follows:
Mine 1  £5 million
Mine 2  £4 million
Mine 3  £4 million
Mine 4  £5 million
There is an upper limit to the amount of ore, which can be extracted from
each mine in a year. These upper limits are as follows:
Mine 1  2 × 10^6 tons
Mine 2  2.5 × 10^6 tons
Mine 3  1.3 × 10^6 tons
Mine 4  3 × 10^6 tons

The ore from the different mines is of varying quality. This quality is mea-
sured on a scale so that blending ores together results in a linear combination
of the quality measurements, for example, if equal quantities of two ores were
combined, the resultant ore would have a quality measurement half way between
that of the ingredient ores. Measured in these units the qualities of the ores from
the mines are given as follows:
Mine 1  1.0
Mine 2  0.7
Mine 3  1.5
Mine 4  0.5
In each year, it is necessary to combine the total outputs from each mine to
produce a blended ore of exactly some stipulated quality. For each year, these
qualities are as follows:
Year 1  0.9
Year 2  0.8
Year 3  1.2
Year 4  0.6
Year 5  1.0
The final blended ore sells for £10 ton each year. Revenue and expenditure
for future years must be discounted at a rate of 10% per annum.
Which mines should be operated each year and how much should they
produce?
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo


def _add_sets(model):
    model.mines = pyo.Set(initialize=["Mine 1", "Mine 2", "Mine 3", "Mine 4"])
    model.years = pyo.RangeSet(1, 5)


def _add_params(model):
    model.operating_mines_upper = pyo.Param(initialize=3)
    model.mine_royalty_per_year = pyo.Param(
        model.mines,
        initialize={
            "Mine 1": 5,
            "Mine 2": 4,
            "Mine 3": 4,
            "Mine 4": 5,
        },
    )

    model.mine_ore_production_upper = pyo.Param(
        model.mines,
        initialize={
            "Mine 1": 2,
            "Mine 2": 2.5,
            "Mine 3": 1.3,
            "Mine 4": 3,
        },
    )

    model.ore_quality = pyo.Param(
        model.mines,
        initialize={
            "Mine 1": 1.0,
            "Mine 2": 0.7,
            "Mine 3": 1.5,
            "Mine 4": 0.5,
        },
    )

    model.ore_quality_required = pyo.Param(
        model.years,
        initialize={
            1: 0.9,
            2: 0.8,
            3: 1.2,
            4: 0.6,
            5: 1.0,
        },
    )

    model.discount = pyo.Param(model.years, initialize=lambda m, y: 1 / 1.1 ** (y - 1))

    model.ore_revenue_per_tonne = pyo.Param(
        model.years, initialize=lambda m, y: 10 * m.discount[y]
    )


def _add_variables(model):
    model.mine_worked = pyo.Var(
        model.mines,
        model.years,
        domain=pyo.Binary,
    )

    model.mine_open = pyo.Var(
        model.mines,
        model.years,
        domain=pyo.Binary,
    )

    model.mine_production = pyo.Var(
        model.mines,
        model.years,
        domain=pyo.NonNegativeReals,
    )

    model.blended_ore_production = pyo.Var(
        model.years,
        domain=pyo.NonNegativeReals,
    )


def _mine_constraints(model):
    # Big M link binary variable to upper constraint of production
    model.mine_production_link = pyo.Constraint(
        model.mines,
        model.years,
        rule=lambda _m, mine, year: _m.mine_production[mine, year]
        <= _m.mine_ore_production_upper[mine] * _m.mine_worked[mine, year],
    )

    # Max 3 mines worked per year
    model.mine_worked_upper = pyo.Constraint(
        model.years,
        rule=lambda _m, year: sum(_m.mine_worked[mine, year] for mine in _m.mines)
        <= _m.operating_mines_upper,
    )

    model.closed_mine_cant_be_worked = pyo.Constraint(
        model.mines,
        model.years,
        rule=lambda _m, mine, year: _m.mine_worked[mine, year]
        - _m.mine_open[mine, year]
        <= 0,
    )

    def _mine_closed_permanently_rule(_m, mine, year):
        if year == _m.years.first():
            return pyo.Constraint.Skip
        return _m.mine_open[mine, year] <= _m.mine_open[mine, _m.years.prev(year)]

    model.mine_closed_on_subsequent_years = pyo.Constraint(
        model.mines, model.years, rule=_mine_closed_permanently_rule
    )


def _ore_constraints(model):
    def _ore_blend_rule(_m, year):
        return (
            sum(
                _m.mine_production[mine, year] * _m.ore_quality[mine]
                for mine in _m.mines
            )
            == _m.blended_ore_production[year] * _m.ore_quality_required[year]
        )

    model.ore_blend = pyo.Constraint(model.years, rule=_ore_blend_rule)

    def _ore_volume_rule(_m, year):
        return (
            sum(_m.mine_production[mine, year] for mine in _m.mines)
            == _m.blended_ore_production[year]
        )

    model.ore_volume = pyo.Constraint(model.years, rule=_ore_volume_rule)


def _add_constraints(model):
    _mine_constraints(model)
    _ore_constraints(model)


def _add_expressions(model):
    model.revenue_by_year = pyo.Expression(
        model.years,
        rule=lambda _m, year: _m.blended_ore_production[year]
        * _m.ore_revenue_per_tonne[year],
    )

    def _royalties(_m, year):
        return _m.discount[year] * sum(
            _m.mine_open[mine, year] * _m.mine_royalty_per_year[mine]
            for mine in model.mines
        )

    model.royalties_per_year = pyo.Expression(
        model.years,
        rule=_royalties,
    )


def _set_objective(model):
    model.OBJ = pyo.Objective(
        expr=sum(
            model.revenue_by_year[year] - model.royalties_per_year[year]
            for year in model.years
        ),
        sense=pyo.maximize,
    )


def _ore_production_plot(model, ax):
    production = model.mine_production.extract_values()
    years = model.years.data()
    production_transformed: dict[str, list[float]] = defaultdict(list)
    for (mine, year), prod in production.items():
        production_transformed[mine].insert(year - 1, prod)

    width = 0.5
    bottom = np.zeros(5)

    for mine, prod in production_transformed.items():
        ax.bar(years, prod, width=width, label=mine, bottom=bottom)
        bottom += prod

    ax.set_title("Ore Production")
    ax.legend(loc="upper right")


def _ore_production_as_pct_of_mine_capacity(model, ax):
    production = model.mine_production.extract_values()
    mine_capacity = model.mine_ore_production_upper.extract_values()
    years = list(model.years.data())
    mines = list(model.mines.data())
    production_pct = np.zeros(
        (
            len(mines),
            len(years),
        )
    )
    for (mine, year), prod in production.items():
        production_pct[mines.index(mine), years.index(year)] = (
            prod / mine_capacity[mine]
        )

    ax.imshow(production_pct, aspect="auto")
    ax.set_xticks(
        range(len(years)),
        labels=years,
    )
    ax.set_yticks(range(len(mines)), labels=mines)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mines)):
        for j in range(len(years)):
            ax.text(
                j,
                i,
                round(production_pct[i, j], 2),
                ha="center",
                va="center",
                color="w",
            )
    ax.set_title("Production as % of capacity")


def _profit_plot(model, ax):
    years = model.years.data()
    revenue, royalties = [], []
    for year in years:
        revenue.append(pyo.value(model.revenue_by_year[year]))
        royalties.append(pyo.value(model.royalties_per_year[year]))
    profit = np.array(revenue) - np.array(royalties)

    ax.plot(years, revenue, marker="o", label="Revenue")
    ax.plot(years, royalties, marker="o", label="Royalties")
    ax.plot(years, profit, marker="o", label="Profit")

    ax.set_xlabel("Year")
    ax.set_ylabel("£M")
    ax.legend()
    ax.grid(True, alpha=0.3)


def report(model):
    fig, ax = plt.subplots(2, 2)

    # 1 stacked bar chart of ore used
    _ore_production_plot(model, ax[0][0])
    _ore_production_as_pct_of_mine_capacity(model, ax[0][1])
    # 1 stacked bar chart of money
    _profit_plot(model, ax[1][0])
    fig.tight_layout()
    plt.show()


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="mining")
    _add_sets(model)
    _add_params(model)
    _add_variables(model)
    _add_constraints(model)
    _add_expressions(model)
    _set_objective(model)
    results = solver.solve(model)
    print(results)
    print(model.display())
    report(model)


if __name__ == "__main__":
    main()
