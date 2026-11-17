"""
12.8 Farm planning
A farmer wishes to plan production on his 200 acre farm over the next five years.
At present, he has a herd of 120 cows. This is made up of 20 heifers and 100
milk-producing cows. Each heifer needs 2/3 acre to support it and each dairy
cow 1 acre. A dairy cow produces an average of 1.1 calves per year. Half of
these calves will be bullocks, which are sold almost immediately for an average
of £30 each. The remaining heifers can be either sold almost immediately for £40
or reared to become milk-producing cows at two years old. It is intended that all
dairy cows be sold at 12 years old for an average of £120 each, although there
will probably be an annual loss of 5% per year among heifers and 2% among
dairy cows. At present, there are 10 cows each aged from newborn to 11 years
old. The decision of how many heifers to sell in the current year has already
been taken and implemented.
The milk from a cow yields an annual revenue of £370. A maximum of 130
cows can be housed at the present time. To provide accommodation for each
cow beyond this number will entail a capital outlay of £200 per cow. Each milk-
producing cow requires 0.6 tons of grain and 0.7 tons of sugar beet per year.
Grain and sugar beet can both be grown on the farm. Each acre yields 1.5 tons
of sugar beet. Only 80 acres are suitable for growing grain. They can be divided
into four groups whose yields are as follows:

Group 1 20 acres 1.1 tons per acre
Group 2 30 acres 0.9 tons per acre
Group 3 20 acres 0.8 tons per acre
Group 4 10 acres 0.65 tons per acre

Grain can be bought for £90 per ton and sold for £75 per ton. Sugar beet can
be bought for £70 per ton and sold for £58 per ton.
The labour requirements are as follows:

Each heifer 10 h per year
Each milk-producing cow 42 h per year
Each acre put to grain 4 h per year
Each acre put to sugar beet 14 h per year

Other costs are as follows:

Each heifer £50 per year
Each milk-producing cow £100 per year
Each acre put to grain £15 per year
Each acre put to sugar beet £10 per year

Labour costs for the farm are at present £4000 per year and provide 5500 h
labour. Any labour needed above this will cost £1.20 per hour.

How should the farmer operate over the next five years to maximise profit?

Any capital expenditure would be financed by a 10-year loan at 15% annual
interest. The interest and capital repayment would be paid in 10 equally sized
yearly instalments. In no year can the cash flow be negative. Finally, the farmer
would neither wish to reduce the total number of dairy cows at the end of the
five-year period by more than 50% nor wish to increase the number by more
than 75%.
"""

import pyomo.environ as pyo
from pyomo.environ import units as u
from pyomo.util.check_units import assert_units_consistent


u.load_definitions_from_strings(
    [
        "GBP = [currency]",
        "cow = [cow]",
    ]
)


def _add_sets(model):
    model.crops = pyo.Set(initialize=["grain", "sugar beet"])
    model.grain_land_groups = pyo.RangeSet(4)
    model.years = pyo.RangeSet(5)
    model.cow_age = pyo.RangeSet(12)


def _add_farm_details(model):
    model.farm_size = pyo.Param(initialize=200, units=u.acre)
    model.labour_cost = pyo.Param(initialize=4000, units=u.GBP)
    model.labour_adt_cost = pyo.Param(initialize=1.2, units=u.GBP)
    model.labour_limit = pyo.Param(initialize=5500, units=u.hour)


def _add_crop_details(model):
    # params
    model.grain_land_group_acre = pyo.Param(
        model.grain_land_groups,
        initialize={1: 20, 2: 30, 3: 20, 4: 10},
        units=u.acre,
    )
    model.grain_land_group_yield = pyo.Param(
        model.grain_land_groups,
        initialize={1: 1.1, 2: 0.9, 3: 0.8, 4: 0.65},
        units=u.tonne / u.acre,
    )
    model.sugar_beet_yield = pyo.Param(
        initialize=1.5,
        units=u.tonne / u.acre,
    )
    model.crop_price_per_tonne = pyo.Param(
        model.crops,
        initialize={"grain": 90, "sugar beet": 70},
        units=u.GBP,
    )
    model.crop_revenue_per_tonne = pyo.Param(
        model.crops,
        initialize={"grain": 75, "sugar beet": 58},
        units=u.GBP,
    )
    model.crop_labour_cost_per_acre = pyo.Param(
        model.crops,
        initialize={"grain": 4, "sugar beet": 14},
        units=u.hour,
    )
    model.crop_cost_per_acre = pyo.Param(
        model.crops,
        initialize={"grain": 15, "sugar beet": 10},
        units=u.GBP,
    )

    # variables
    model.grain_grown_per_group = pyo.Var(
        model.grain_land_groups,
        model.years,
        domain=pyo.NonNegativeReals,
        bounds=lambda _m, grain_group, year: (0, _m.grain_land_group_acre[grain_group] *
                                  _m.grain_land_group_yield[grain_group]),
        units=u.tonne,
    )

    model.sugar_beet_grown = pyo.Var(
        model.years,
        domain=pyo.NonNegativeReals,
        bounds=(0, model.farm_size * model.sugar_beet_yield),
        units=u.tonne,
    )

    model.grain_bought = pyo.Var(
        model.years,
        domain=pyo.NonNegativeReals,
        units=u.tonne,
    )

    model.grain_sold = pyo.Var(
        model.years,
        domain=pyo.NonNegativeReals,
        units=u.tonne,
    )

    model.sugar_beet_bought = pyo.Var(
        model.years,
        domain=pyo.NonNegativeReals,
        units=u.tonne,
    )

    model.sugar_beet_sold = pyo.Var(
        model.years,
        domain=pyo.NonNegativeReals,
        units=u.tonne,
    )

def _add_cow_details(model):
    model.heifer_acre_requirement = pyo.Param(initialize=2/3, units=u.acre)
    model.calf_production = pyo.Param(initialize=1.1, units=u.cow)
    model.bullock_production = pyo.Param(initialize=model.calf_production *
                                                   0.5, units=u.cow)
    model.bullock_price = pyo.Param(initialize=30, units=u.GBP)
    model.heifer_production = pyo.Param(initialize=model.calf_production *
                                                   0.5, units=u.cow)
    model.heifer_price = pyo.Param(initialize=40, units=u.GBP)
    model.heifer_loss = pyo.Param(initialize=0.05)

    model.heifer_to_cow_year = pyo.Param(initialize=3)
    model.cow_acre_requirement = pyo.Param(initialize=1, units=u.acre)
    model.cow_price = pyo.Param(initialize=120, units=u.GBP)

    model.cow_loss = pyo.Param(initialize=0.02)


    # variables
    model.cows = pyo.Var(
        pyo.cow_age, pyo.years,
        domain=pyo.NonNegativeIntegers,
    )

def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="farm_planning")
    _add_sets(model)
    _add_farm_details(model)
    _add_crop_details(model)

    assert_units_consistent(model)
    # results = solver.solve(model)


if __name__ == "__main__":
    main()
