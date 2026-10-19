"""
12.13 Market sharing
A large company has two divisions, D1 and D2. The company supplies retail-
ers with oil and spirit. This is a much smaller version of the problem British
Petroleum and Shell faced when they were forced to demerge – one of the largest
demergers in history. The original model proved impossible to solve in 1972.
It is desired to allocate each retailer to either division D1 or division D2.
This division will be the retailer’s supplier. As far as possible, this division must
be made so that D1 controls 40% of the market and D2 the remaining 60%. The
retailers are listed below as M1 to M23. Each retailer has an estimated market
for oil and spirit. Retailers M1 to M8 are in region 1; retailers M9 to M18
are in region 2 and retailers M19 to M23 are in region 3. Certain retailers are
considered to have good growth prospects and categorised as group A and the
others are in group B. Each retailer has a certain number of delivery points as
given below. It is desired to make the 40/60 split between D1 and D2 in each
of the following respects:
1. Total number of delivery points
2. Control of spirit market
3. Control of oil market in region 1
4. Control of oil market in region 2
5. Control of oil market in region 3
6. Number of retailers in group A
7. Number of retailers in group B.
There is a certain flexibility in that any share may vary by ±5%. That is, the
share can vary between the limits 35/65 and 45/55.
The primary aim is to find a feasible solution. If, however, there is some
choice then possible objectives are (i) to minimise the sum of the percentage
deviations from the 40/60 split and (ii) to minimise the maximum such deviation.
Build a model to see if the problem has a feasible solution and if so find the
optimal solutions.
The numerical data are given in Table 12.5.
Table 12.5
            Retailer    Oil market  Delivery points Spirit market   Growth category
Region 1    M1          9           11              34              A
            M2          13          47              411             A
            M3          14          44              82              A
            M4          17          25              157             B
            M5          18          10              5               A
            M6          19          26              183             A
            M7          23          26              14              B
            M8          21          54              215             B
Region 2    M9          9           18              102             B
            M10         11          51              21              A
            M11         17          20              54              B
            M12         18          105             0               B
            M13         18          7               6               B
            M14         17          16              96              B
            M15         22          34              118             A
            M16         24          100             112             B
            M17         36          50              535             B
            M18         43          21              8               B
Region 3    M19         6           11              53              B
            M20         15          19              28              A
            M21         15          14              69              B
            M22         25          10              65              B
            M23         39          11              27              B
"""

import pandas as pd
import pyomo.environ as pyo

DIVISIONS = ["D1", "D2"]


RETAILERS = [f"M{i}" for i in range(1, 24)]

DATA = pd.DataFrame(
    data=[
        [9, 11, 34, "A", "Region 1"],
        [13, 47, 411, "A", "Region 1"],
        [14, 44, 82, "A", "Region 1"],
        [17, 25, 157, "B", "Region 1"],
        [18, 10, 5, "A", "Region 1"],
        [19, 26, 183, "A", "Region 1"],
        [23, 26, 14, "B", "Region 1"],
        [21, 54, 215, "B", "Region 1"],
        [9, 18, 102, "B", "Region 2"],
        [11, 51, 21, "A", "Region 2"],
        [17, 20, 54, "B", "Region 2"],
        [18, 105, 0, "B", "Region 2"],
        [18, 7, 6, "B", "Region 2"],
        [17, 16, 96, "B", "Region 2"],
        [22, 34, 118, "A", "Region 2"],
        [24, 100, 112, "B", "Region 2"],
        [36, 50, 535, "B", "Region 2"],
        [43, 21, 8, "B", "Region 2"],
        [6, 11, 53, "B", "Region 3"],
        [15, 19, 28, "A", "Region 3"],
        [15, 14, 69, "B", "Region 3"],
        [25, 10, 65, "B", "Region 3"],
        [39, 11, 27, "B", "Region 3"],
    ],
    index=RETAILERS,
    columns=[
        "Oil Market",
        "Delivery points",
        "Spirit Market",
        "Growth Category",
        "Region",
    ],
)

CRITERIA = [
    "delivery_points",
    "spirit_market",
    "oil_region_1",
    "oil_region_2",
    "oil_region_3",
    "group_A",
    "group_B",
]


def _add_sets(model):
    model.divisions = pyo.Set(initialize=DIVISIONS)
    model.retailers = pyo.Set(initialize=RETAILERS)

    model.criteria = pyo.Set(initialize=CRITERIA)  # or named criteria


def _add_params(model):
    model.oil_market = pyo.Param(
        model.retailers,
        initialize=DATA["Oil Market"].to_dict(),
    )

    model.delivery_points = pyo.Param(
        model.retailers,
        initialize=DATA["Delivery points"].to_dict(),
    )

    model.spirit_market = pyo.Param(
        model.retailers,
        initialize=DATA["Spirit Market"].to_dict(),
    )

    model.growth_category = pyo.Param(
        model.retailers,
        within=pyo.Any,
        initialize=DATA["Growth Category"].to_dict(),
    )
    model.region = pyo.Param(
        model.retailers,
        within=pyo.Any,
        initialize=DATA["Region"].to_dict(),
    )

    model.d1_control_target = pyo.Param(initialize=0.4)

    def coeff_rule(m, crit, ret):
        if crit == "delivery_points":
            return m.delivery_points[ret]
        if crit == "spirit_market":
            return m.spirit_market[ret]
        if crit.startswith("oil_region_"):
            region = "Region " + crit[-1]
            return m.oil_market[ret] if m.region[ret] == region else 0
        if crit.startswith("group_"):
            return 1 if m.growth_category[ret] == crit[-1] else 0
        raise ValueError(crit)

    model.coeff = pyo.Param(model.criteria, model.retailers, initialize=coeff_rule)
    model.total = pyo.Param(
        model.criteria,
        initialize=lambda m, k: sum(m.coeff[k, r] for r in m.retailers),
    )


def _add_variables(model):
    model.retailer_allocation = pyo.Var(
        model.retailers,
        domain=pyo.Binary,
    )
    model.dev = pyo.Var(model.criteria, bounds=(0, 0.05))
    model.max_dev = pyo.Var(bounds=(0, 0.05))


def _add_constraints(model):
    def _d1_share(_m, crit):
        return (
            sum(
                _m.coeff[crit, ret] * _m.retailer_allocation[ret]
                for ret in model.retailers
            )
            / _m.total[crit]
        )

    model.dev_upper = pyo.Constraint(
        model.criteria,
        rule=lambda _m, crit: _d1_share(_m, crit) - _m.d1_control_target
        <= _m.dev[crit],
    )

    model.dev_lower = pyo.Constraint(
        model.criteria,
        rule=lambda _m, crit: _m.d1_control_target - _d1_share(_m, crit)
        <= _m.dev[crit],
    )

    model.max_dev_link = pyo.Constraint(
        model.criteria,
        rule=lambda m, k: m.dev[k] <= m.max_dev,
    )


def _add_objectives(model):
    model.obj_sum = pyo.Objective(
        expr=sum(model.dev[k] for k in model.criteria), sense=pyo.minimize
    )
    model.obj_max = pyo.Objective(expr=model.max_dev, sense=pyo.minimize)
    model.obj_max.deactivate()  # swap with obj_sum.deactivate() to switch


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="market_sharing")
    _add_sets(model)
    _add_params(model)
    _add_variables(model)
    _add_constraints(model)
    _add_objectives(model)
    results = solver.solve(model)
    tc = results.solver.termination_condition
    print("Termination:", tc)
    if tc != pyo.TerminationCondition.optimal:
        return

    d1 = [r for r in model.retailers if pyo.value(model.retailer_allocation[r]) > 0.5]
    print("D1 retailers:", d1)
    for k in model.criteria:
        share = sum(model.coeff[k, r] for r in d1) / model.total[k]
        print(f"{k:16s} D1 share = {share:.3f}")
    print("Objective:", pyo.value(model.obj_sum))


if __name__ == "__main__":
    main()
