"""
12.6 Refinery optimisation
An oil refinery purchases two crude oils (crude 1 and crude 2). These crude oils
are put through four processes: distillation, reforming, cracking and blending, to
produce petrols and fuels that are sold.
12.6.1 Distillation
Distillation separates each crude oil into fractions known as light naphtha,
medium naphtha, heavy naphtha, light oil, heavy oil and residuum according to
their boiling points. Light, medium and heavy naphthas have octane numbers of
90, 80 and 70, respectively. The fractions into which one barrel of each type of
crude splits are given in the following table:
        Light naphtha   Medium naphtha  Heavy naphtha   Light oil   Heavy oil   Residuum
Crude 1 0.1             0.2             0.2             0.12        0.2         0.13
Crude 2 0.15            0.25            0.18            0.08        0.19        0.12
N.B. There is a small amount of wastage in distillation.
12.6.2 Reforming
The naphthas can be used immediately for blending into different grades of
petrol or can go through a process known as reforming. Reforming produces a
product known as reformed gasoline with an octane number of 115. The yields
of reformed gasoline from each barrel of the different naphthas are given as
follows:
1 barrel of light naphtha yields 0.6 barrels of reformed gasoline;
1 barrel of medium naphtha yields 0.52 barrels of reformed gasoline;
1 barrel of heavy naphtha yields 0.45 barrels of reformed gasoline.
12.6.3 Cracking
The oils (light and heavy) can either be used directly for blending into jet fuel or
fuel oil or be put through a process known as catalytic cracking. The catalytic
cracker produces cracked oil and cracked gasoline. Cracked gasoline has an
octane number of 105.
1 barrel of light oil yields 0.68 barrels of cracked oil and 0.28 barrels of
cracked gasoline;
1 barrel of heavy oil yields 0.75 barrels of cracked oil and 0.2 barrels of
cracked gasoline.
Cracked oil is used for blending fuel oil and jet fuel ; cracked gasoline is used
for blending petrol.
Residuum can be used for either producing lube-oil or blending into jet fuel
and fuel oil :
1 barrel of residuum yields 0.5 barrels of lube-oil.
12.6.4 Blending
12.6.4.1 Petrols (motor fuel)
There are two sorts of petrol, regular and premium, obtained by blending the
naphtha, reformed gasoline and cracked gasoline. The only stipulations concern-
ing them are that regular must have an octane number of at least 84 and that
premium must have an octane number of at least 94. It is assumed that octane
numbers blend linearly by volume.
12.6.4.2 Jet fuel
The stipulation concerning jet fuel is that its vapour pressure must not exceed
1 kg cm2. The vapour pressures for light, heavy, cracked oils and residuum are
1.0, 0.6, 1.5 and 0.05 kg cm2, respectively. It may again be assumed that vapour
pressures blend linearly by volume.
12.6.4.3 Fuel oil
To produce fuel oil, light oil, cracked oil, heavy oil and residuum must be blended
in the ratio 10:4:3:1.
There are availability and capacity limitations on the quantities and processes
used as follows:
1. The daily availability of crude 1 is 20 000 barrels.
2. The daily availability of crude 2 is 30 000 barrels.
3. At most 45 000 barrels of crude can be distilled per day.
4. At most 10 000 barrels of naphtha can be reformed per day.
5. At most 8000 barrels of oil can be cracked per day.
6. The daily production of lube oil must be between 500 and 1000 barrels.
7. Premium motor fuel production must be at least 40% of regular motor fuel
production.
barrel) as follows:
The profit contributions from the sale of the final products are (in pence per
Premium petrol  700
Regular petrol  600
Jet fuel        400
Fuel oil        350
Lube-oil        150
How should the operations of the refinery be planned in order to maximise
total profit?
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port


def _crude(crude) -> None:
    """Create the crude oil flow block"""
    crude.oils = pyo.Set(initialize=["CRUDE 1", "CRUDE 2"])
    crude.availability = pyo.Param(
        crude.oils, initialize=dict(zip(crude.oils, [20_000, 30_000]))
    )
    crude.flow = pyo.Var(
        crude.oils,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda _crude, oil: (0, crude.availability[oil]),
    )
    crude.max_flow = pyo.Constraint(
        expr=sum(crude.flow[oil] for oil in crude.oils) <= 45_000
    )
    crude.outlet = Port()
    crude.outlet.add(crude.flow, "crude_flow")
    return crude


def _distillation(distill) -> None:
    """Create the distillation Block"""
    model = distill.model()
    distill.naphtha = pyo.Set(
        initialize=["Light Naphtha", "Medium Naphtha", "Heavy Naphtha"]
    )
    distill.naphtha_octane = pyo.Param(
        distill.naphtha,
        initialize=dict(zip(distill.naphtha, [90, 80, 70])),
    )
    distill.oil = pyo.Set(initialize=["Light Oil", "Heavy Oil"])
    distill.residuum = pyo.Set(initialize=["Residuum"])
    distill.vapour_pressure = pyo.Param(
        distill.oil | distill.residuum,
        initialize=dict(zip(distill.oil | distill.residuum, [1.0, 0.6, 0.05])),
    )
    distill.product = distill.naphtha | distill.oil | distill.residuum

    # crude input flow
    distill.inlet_flow = pyo.Var(
        model.crude.oils,
        domain=pyo.NonNegativeIntegers,
    )
    distill.inlet = Port()
    distill.inlet.add(distill.inlet_flow, "crude_flow")

    # all product flow and fraction of crude
    distill.product_flow = pyo.Var(
        distill.product,
        domain=pyo.NonNegativeIntegers,
    )

    distill.product_yield = pyo.Param(
        model.crude.oils,
        distill.product,
        initialize={
            ("CRUDE 1", "Light Naphtha"): 0.1,
            ("CRUDE 1", "Medium Naphtha"): 0.2,
            ("CRUDE 1", "Heavy Naphtha"): 0.2,
            ("CRUDE 1", "Light Oil"): 0.12,
            ("CRUDE 1", "Heavy Oil"): 0.2,
            ("CRUDE 1", "Residuum"): 0.13,
            ("CRUDE 2", "Light Naphtha"): 0.15,
            ("CRUDE 2", "Medium Naphtha"): 0.25,
            ("CRUDE 2", "Heavy Naphtha"): 0.18,
            ("CRUDE 2", "Light Oil"): 0.08,
            ("CRUDE 2", "Heavy Oil"): 0.19,
            ("CRUDE 2", "Residuum"): 0.12,
        },
    )

    def _distillation_rule(b, product):
        return b.product_flow[product] == sum(
            b.inlet_flow[crude] * b.product_yield[crude, product]
            for crude in model.crude.oils
        )

    distill.product_conversion = pyo.Constraint(
        distill.product,
        rule=_distillation_rule,
    )

    # naphtha flow
    distill.naphtha_flow = pyo.Var(
        distill.naphtha,
        domain=pyo.NonNegativeIntegers,
    )

    def _naphtha_flow_rule(b, product):
        return b.naphtha_flow[product] == b.product_flow[product]

    distill.naphtha_interface = pyo.Constraint(
        distill.naphtha,
        rule=_naphtha_flow_rule,
    )
    distill.naphtha_outlet = Port()
    distill.naphtha_outlet.add(
        distill.naphtha_flow,
        "naphtha_flow",
        rule=Port.Extensive,
        include_splitfrac=False,
    )

    # oils flow
    distill.oil_outlet = Port()

    distill.oil_flow = pyo.Var(
        distill.oil,
        domain=pyo.NonNegativeIntegers,
    )

    def _oil_flow_rule(b, product):
        return b.oil_flow[product] == b.product_flow[product]

    distill.oil_interface = pyo.Constraint(
        distill.oil,
        rule=_oil_flow_rule,
    )
    distill.oil_outlet.add(
        distill.oil_flow,
        "oil_flow",
        rule=Port.Extensive,
        include_splitfrac=False,
    )

    # residuum flow
    distill.residuum_flow = pyo.Var(
        domain=pyo.NonNegativeIntegers,
    )

    def _residuum_flow_rule(b, product):
        return b.residuum_flow == b.product_flow["Residuum"]

    distill.residuum_interface = pyo.Constraint(
        distill.residuum,
        rule=_residuum_flow_rule,
    )
    distill.residuum_outlet = Port()
    distill.residuum_outlet.add(
        distill.residuum_flow,
        "residuum_flow",
        rule=Port.Extensive,
        include_splitfrac=False,
    )
    return distill


def _reforming(reform) -> None:
    model = reform.model()
    reform.product = pyo.Set(initialize=["reformed gasoline"])
    reform.gas_octane = pyo.Param(
        reform.product,
        initialize=dict(zip(reform.product, [115])),
    )
    reform.product_yield = pyo.Param(
        model.distillation.naphtha,
        initialize=dict(zip(model.distillation.naphtha, [0.6, 0.52, 0.45])),
    )
    reform.naphtha_upper = pyo.Param(initialize=10_000)

    # naphtha input flow
    reform.naphtha_flow = pyo.Var(
        model.distillation.naphtha,
        domain=pyo.NonNegativeIntegers,
    )
    reform.naphtha_limit = pyo.Constraint(
        expr=sum(reform.naphtha_flow[naphtha] for naphtha in model.distillation.naphtha)
        <= reform.naphtha_upper
    )
    reform.inlet = Port()
    reform.inlet.add(reform.naphtha_flow, "naphtha_flow", rule=Port.Extensive)

    # outflow
    reform.product_flow = pyo.Var(
        reform.product,
        domain=pyo.NonNegativeIntegers,
    )

    def _yield_rule(b, product):
        return b.product_flow[product] == sum(
            b.naphtha_flow[naphtha] * b.product_yield[naphtha]
            for naphtha in model.distillation.naphtha
        )

    reform.product_conversion = pyo.Constraint(
        reform.product,
        rule=_yield_rule,
    )

    reform.outlet = Port()
    reform.outlet.add(reform.product_flow, "reform_flow")


def _cracking(cracking) -> None:
    model = cracking.model()
    # products
    cracking.gas = pyo.Set(initialize=["cracked gasoline"])
    cracking.oil = pyo.Set(initialize=["cracked oil"])
    cracking.product = cracking.gas | cracking.oil

    # product params
    cracking.gas_octane = pyo.Param(
        cracking.gas,
        initialize=dict(zip(cracking.gas, [105])),
    )
    cracking.vapour_pressure = pyo.Param(
        cracking.oil, initialize=dict(zip(cracking.oil, [1.5]))
    )
    cracking.product_yield = pyo.Param(
        model.distillation.oil,
        cracking.product,
        initialize={
            ("Light Oil", "cracked gasoline"): 0.28,
            ("Light Oil", "cracked oil"): 0.68,
            ("Heavy Oil", "cracked gasoline"): 0.2,
            ("Heavy Oil", "cracked oil"): 0.75,
        },
    )
    cracking.oil_in_upper = pyo.Param(initialize=8000)

    # inflow
    cracking.oil_inflow = pyo.Var(
        model.distillation.oil,
        domain=pyo.NonNegativeIntegers,
    )
    cracking.inlet = Port()
    cracking.inlet.add(cracking.oil_inflow, "oil_flow", rule=Port.Extensive)

    cracking.oil_in_limit = pyo.Constraint(
        expr=sum(cracking.oil_inflow[oil] for oil in model.distillation.oil)
        <= cracking.oil_in_upper
    )
    # outflow
    cracking.product_flow = pyo.Var(
        cracking.product,
        domain=pyo.NonNegativeIntegers,
    )

    def _yield_rule(b, product):
        return b.product_flow[product] == sum(
            b.oil_inflow[oil] * b.product_yield[oil, product]
            for oil in model.distillation.oil
        )

    cracking.product_conversion = pyo.Constraint(
        cracking.product,
        rule=_yield_rule,
    )

    # oil flow
    cracking.oil_outflow = pyo.Var(
        cracking.oil,
        domain=pyo.NonNegativeIntegers,
    )

    def _oil_flow_rule(b, product):
        return b.oil_outflow[product] == b.product_flow[product]

    cracking.oil_interface = pyo.Constraint(
        cracking.oil,
        rule=_oil_flow_rule,
    )

    cracking.oil_outlet = Port()
    cracking.oil_outlet.add(
        cracking.oil_outflow,
        "cracked_oil_flow",
        rule=Port.Extensive,
        include_splitfrac=False,
    )

    # gasoline flow
    cracking.gas_outflow = pyo.Var(
        cracking.gas,
        domain=pyo.NonNegativeIntegers,
    )

    def _gas_flow_rule(b, product):
        return b.gas_outflow[product] == b.product_flow[product]

    cracking.gas_interface = pyo.Constraint(
        cracking.gas,
        rule=_gas_flow_rule,
    )
    cracking.gas_outlet = Port()
    cracking.gas_outlet.add(cracking.gas_outflow, "cracked_gas_flow")


def _petrol(petrol) -> None:
    # sets
    petrol.types = pyo.Set(initialize=["Regular", "Premium"])
    # params
    petrol.octane_lower = pyo.Param(
        petrol.types, initialize=dict(zip(petrol.types, [84, 94]))
    )
    petrol.profit_contribution = pyo.Param(
        petrol.types, initialize=dict(zip(petrol.types, [600, 700]))
    )

    # inflows
    #   naphtha
    model = petrol.model()
    petrol.naphtha_available = pyo.Var(
        model.distillation.naphtha, domain=pyo.NonNegativeIntegers
    )

    petrol.reformed_gas_available = pyo.Var(
        model.reforming.product, domain=pyo.NonNegativeIntegers
    )

    petrol.cracked_gas_available = pyo.Var(
        model.cracking.gas, domain=pyo.NonNegativeIntegers
    )
    petrol.reformed_gas_flow = pyo.Var(
        model.reforming.product,
        domain=pyo.NonNegativeIntegers,
    )
    petrol.naphtha_inlet = Port()
    petrol.naphtha_inlet.add(
        petrol.naphtha_available, "naphtha_flow", rule=Port.Extensive
    )

    #   reformed gas
    petrol.reform_inlet = Port()
    petrol.reform_inlet.add(petrol.reformed_gas_available, "reform_flow")
    #   cracked gas

    petrol.cracked_inlet = Port()
    petrol.cracked_inlet.add(petrol.cracked_gas_available, "cracked_gas_flow")

    # ---------------------------------------------------------
    # Split incoming material between Regular and Premium
    # ---------------------------------------------------------

    petrol.naphtha_to_grade = pyo.Var(
        petrol.types, model.distillation.naphtha, domain=pyo.NonNegativeIntegers
    )

    petrol.reformed_gas_to_grade = pyo.Var(
        petrol.types, model.reforming.product, domain=pyo.NonNegativeIntegers
    )

    petrol.cracked_gas_to_grade = pyo.Var(
        petrol.types, model.cracking.gas, domain=pyo.NonNegativeIntegers
    )

    # All incoming naphtha must be allocated to either
    # Regular or Premium.

    petrol.naphtha_split = pyo.Constraint(
        model.distillation.naphtha,
        rule=lambda m, n: (
            sum(m.naphtha_to_grade[p, n] for p in m.types) == m.naphtha_available[n]
        ),
    )

    # All reformed gasoline goes to either Regular or Premium.

    petrol.reformed_gas_split = pyo.Constraint(
        model.reforming.product,
        rule=lambda m, r: (
            sum(m.reformed_gas_to_grade[p, r] for p in m.types)
            == m.reformed_gas_available[r]
        ),
    )

    # All cracked gasoline goes to either Regular or Premium.

    petrol.cracked_gas_split = pyo.Constraint(
        model.cracking.gas,
        rule=lambda m, c: (
            sum(m.cracked_gas_to_grade[p, c] for p in m.types)
            == m.cracked_gas_available[c]
        ),
    )
    # production
    petrol.production = pyo.Var(
        petrol.types,
        domain=pyo.NonNegativeIntegers,
    )

    def _lower_octane_rule(_b, p_type):
        nap = sum(
            _b.naphtha_to_grade[p_type, nap] * model.distillation.naphtha_octane[nap]
            for nap in model.distillation.naphtha
        )
        reform = sum(
            _b.reformed_gas_to_grade[p_type, rg] * model.reforming.gas_octane[rg]
            for rg in model.reforming.product
        )
        cracked = sum(
            _b.cracked_gas_to_grade[p_type, cg] * model.cracking.gas_octane[cg]
            for cg in model.cracking.gas
        )
        return nap + reform + cracked >= _b.production[p_type] * _b.octane_lower[p_type]

    petrol.octane_minimum = pyo.Constraint(
        petrol.types,
        rule=_lower_octane_rule,
    )

    def _volume_rule(
        _b,
        p_type,
    ):
        nap = sum(
            _b.naphtha_to_grade[p_type, nap] for nap in model.distillation.naphtha
        )
        reform = sum(
            _b.reformed_gas_to_grade[p_type, rg] for rg in model.reforming.product
        )
        cracked = sum(_b.cracked_gas_to_grade[p_type, cg] for cg in model.cracking.gas)
        return nap + reform + cracked == _b.production[p_type]

    petrol.volume = pyo.Constraint(
        petrol.types,
        rule=_volume_rule,
    )

    petrol.production_balance = pyo.Constraint(
        expr=petrol.production["Premium"] >= 0.4 * petrol.production["Regular"]
    )

    petrol.profit = pyo.Expression(
        expr=sum(
            petrol.production[t] * petrol.profit_contribution[t] for t in petrol.types
        ),
    )


def _jet_fuel(jet_fuel) -> None:
    # params
    jet_fuel.vapor_pressure_upper = pyo.Param(initialize=1.0)
    jet_fuel.profit_contribution = pyo.Param(initialize=400)
    # inflow
    model = jet_fuel.model()
    # distilled oils
    jet_fuel.oil_inlet = Port()
    jet_fuel.oil_flow = pyo.Var(
        model.distillation.oil,
        domain=pyo.NonNegativeIntegers,
    )
    jet_fuel.oil_inlet.add(
        jet_fuel.oil_flow,
        "oil_flow",
        rule=Port.Extensive,
    )
    #   cracked oils
    jet_fuel.cracked_oil_flow = pyo.Var(
        model.cracking.oil,
        domain=pyo.NonNegativeIntegers,
    )
    jet_fuel.cracked_oil_inlet = Port()
    jet_fuel.cracked_oil_inlet.add(
        jet_fuel.cracked_oil_flow,
        "cracked_oil_flow",
        rule=Port.Extensive,
    )
    #   residuum
    jet_fuel.residuum_flow = pyo.Var(
        domain=pyo.NonNegativeIntegers,
    )
    jet_fuel.residuum_inlet = Port()
    jet_fuel.residuum_inlet.add(
        jet_fuel.residuum_flow,
        "residuum_flow",
        rule=Port.Extensive,
    )

    # production
    jet_fuel.production = pyo.Var(
        domain=pyo.NonNegativeIntegers,
    )

    def _upper_vapor_pressure_rule(_b):
        oil = sum(
            _b.oil_flow[oil] * model.distillation.vapour_pressure[oil]
            for oil in model.distillation.oil
        )
        cracked = sum(
            _b.cracked_oil_flow[co] * model.cracking.vapour_pressure[co]
            for co in model.cracking.oil
        )
        res = _b.residuum_flow * model.distillation.vapour_pressure["Residuum"]
        return oil + res + cracked <= _b.production * _b.vapor_pressure_upper

    jet_fuel.vapor_pressure_maximum = pyo.Constraint(
        rule=_upper_vapor_pressure_rule,
    )

    def _volume_rule(_b):
        oil = sum(_b.oil_flow[oil] for oil in model.distillation.oil)
        cracked = sum(_b.cracked_oil_flow[co] for co in model.cracking.oil)
        return oil + _b.residuum_flow + cracked == jet_fuel.production

    jet_fuel.volume = pyo.Constraint(
        rule=_volume_rule,
    )

    jet_fuel.profit = pyo.Expression(
        expr=jet_fuel.production * jet_fuel.profit_contribution
    )


def _fuel_oil(fuel_oil) -> None:
    model = fuel_oil.model()
    # params
    fuel_oil.profit_contribution = pyo.Param(initialize=350)
    fuel_oil.components = (
        model.distillation.oil | model.cracking.oil | model.distillation.residuum
    )
    fuel_oil.ratio = pyo.Param(
        fuel_oil.components,
        initialize={"Light Oil": 10, "Heavy Oil": 3, "Residuum": 1, "cracked oil": 4},
    )

    # inflow
    # distilled oils
    fuel_oil.oil_inlet = Port()
    fuel_oil.oil_flow = pyo.Var(
        model.distillation.oil,
        domain=pyo.NonNegativeIntegers,
    )
    fuel_oil.oil_inlet.add(fuel_oil.oil_flow, "oil_flow", rule=Port.Extensive)
    #   cracked oils
    fuel_oil.cracked_oil_flow = pyo.Var(
        model.cracking.oil,
        domain=pyo.NonNegativeIntegers,
    )
    fuel_oil.cracked_oil_inlet = Port()
    fuel_oil.cracked_oil_inlet.add(
        fuel_oil.cracked_oil_flow, "cracked_oil_flow", rule=Port.Extensive
    )
    #   residuum
    fuel_oil.residuum_flow = pyo.Var(
        domain=pyo.NonNegativeIntegers,
    )
    fuel_oil.residuum_inlet = Port()
    fuel_oil.residuum_inlet.add(
        fuel_oil.residuum_flow,
        "residuum_flow",
        rule=Port.Extensive,
    )

    # production
    fuel_oil.production = pyo.Var(
        domain=pyo.NonNegativeIntegers,
    )

    ratio_total = sum(fuel_oil.ratio[c] for c in fuel_oil.components)

    def _ratio_rule(_b, comp):
        # pick the inflow variable for this component
        if comp in model.distillation.oil:
            flow = _b.oil_flow[comp]
        elif comp in model.cracking.oil:
            flow = _b.cracked_oil_flow[comp]
        else:  # "Residuum"
            flow = _b.residuum_flow
        return flow == _b.production * _b.ratio[comp] / ratio_total

    fuel_oil.ratio_constraint = pyo.Constraint(
        fuel_oil.components,
        rule=_ratio_rule,
    )

    fuel_oil.profit = pyo.Expression(
        expr=fuel_oil.production * fuel_oil.profit_contribution
    )


def _lube_oil(lube_oil) -> None:
    # params
    lube_oil.oil_yield = pyo.Param(initialize=0.5)
    lube_oil.profit_contribution = pyo.Param(initialize=150)

    lube_oil.production = pyo.Var(
        domain=pyo.NonNegativeIntegers,
        bounds=(500, 1000),
    )

    # inlet
    lube_oil.residuum_flow = pyo.Var(
        domain=pyo.NonNegativeIntegers,
    )
    lube_oil.residuum_inlet = Port()
    lube_oil.residuum_inlet.add(
        lube_oil.residuum_flow,
        "residuum_flow",
        rule=Port.Extensive,
    )

    lube_oil.profit = pyo.Expression(
        expr=lube_oil.production * lube_oil.profit_contribution
    )

    lube_oil.yield_constraint = pyo.Constraint(
        expr=lube_oil.production == lube_oil.oil_yield * lube_oil.residuum_flow
    )


def _connect_components(model) -> None:
    model.crude_to_distillation = Arc(
        source=model.crude.outlet,
        destination=model.distillation.inlet,
    )

    model.distillation_to_reforming = Arc(
        source=model.distillation.naphtha_outlet,
        destination=model.reforming.inlet,
    )

    model.distillation_to_cracking = Arc(
        source=model.distillation.oil_outlet,
        destination=model.cracking.inlet,
    )

    model.distillation_to_petrol = Arc(
        source=model.distillation.naphtha_outlet,
        destination=model.petrol.naphtha_inlet,
    )

    model.distillation_oil_to_jet_fuel = Arc(
        source=model.distillation.oil_outlet,
        destination=model.jet_fuel.oil_inlet,
    )

    model.distillation_oil_to_fuel_oil = Arc(
        source=model.distillation.oil_outlet,
        destination=model.fuel_oil.oil_inlet,
    )

    model.residuum_to_jet_fuel = Arc(
        source=model.distillation.residuum_outlet,
        destination=model.jet_fuel.residuum_inlet,
    )

    model.residuum_to_fuel_oil = Arc(
        source=model.distillation.residuum_outlet,
        destination=model.fuel_oil.residuum_inlet,
    )

    model.residuum_to_lube_oil = Arc(
        source=model.distillation.residuum_outlet,
        destination=model.lube_oil.residuum_inlet,
    )

    model.reforming_to_petrol = Arc(
        source=model.reforming.outlet,
        destination=model.petrol.reform_inlet,
    )

    model.cracking_to_petrol = Arc(
        source=model.cracking.gas_outlet,
        destination=model.petrol.cracked_inlet,
    )

    model.cracking_to_jet_fuel = Arc(
        source=model.cracking.oil_outlet,
        destination=model.jet_fuel.cracked_oil_inlet,
    )

    model.cracking_to_fuel_oil = Arc(
        source=model.cracking.oil_outlet,
        destination=model.fuel_oil.cracked_oil_inlet,
    )


def _set_objective(model):
    model.OBJ = pyo.Objective(
        expr=model.petrol.profit
        + model.jet_fuel.profit
        + model.fuel_oil.profit
        + model.lube_oil.profit,
        sense=pyo.maximize,
    )


def report(model) -> None:
    v = pyo.value
    d, rf, cr = model.distillation, model.reforming, model.cracking
    pt, jf, fo, lb = model.petrol, model.jet_fuel, model.fuel_oil, model.lube_oil

    LN, MN, HN = "Light Naphtha", "Medium Naphtha", "Heavy Naphtha"
    LO, HO = "Light Oil", "Heavy Oil"
    RG, CG, CO = "reformed gasoline", "cracked gasoline", "cracked oil"
    PM, RM = "Premium", "Regular"

    rows = [
        # crude
        ("CRA", "crude 1", model.crude.flow["CRUDE 1"]),
        ("CRB", "crude 2", model.crude.flow["CRUDE 2"]),
        # distillation products
        ("LN", "light naphtha", d.naphtha_flow[LN]),
        ("MN", "medium naphtha", d.naphtha_flow[MN]),
        ("HN", "heavy naphtha", d.naphtha_flow[HN]),
        ("LO", "light oil", d.oil_flow[LO]),
        ("HO", "heavy oil", d.oil_flow[HO]),
        ("R", "residuum", d.residuum_flow),
        # reforming
        (
            "LNRG",
            "light naphtha used to produce reformed gasoline",
            rf.naphtha_flow[LN],
        ),
        (
            "MNRG",
            "medium naphtha used to produce reformed gasoline",
            rf.naphtha_flow[MN],
        ),
        (
            "HNRG",
            "heavy naphtha used to produce reformed gasoline",
            rf.naphtha_flow[HN],
        ),
        ("RG", "reformed gasoline", rf.product_flow[RG]),
        # cracking
        (
            "LOCGO",
            "light oil used to produce cracked oil and cracked gasoline",
            cr.oil_inflow[LO],
        ),
        (
            "HOCGO",
            "heavy oil used to produce cracked oil and cracked gasoline",
            cr.oil_inflow[HO],
        ),
        ("CG", "cracked gasoline", cr.gas_outflow[CG]),
        ("CO", "cracked oil", cr.oil_outflow[CO]),
        # petrol blending
        (
            "LNPMF",
            "light naphtha used to produce premium motor fuel",
            pt.naphtha_to_grade[PM, LN],
        ),
        (
            "LNRMF",
            "light naphtha used to produce regular motor fuel",
            pt.naphtha_to_grade[RM, LN],
        ),
        (
            "MNPMF",
            "medium naphtha used to produce premium motor fuel",
            pt.naphtha_to_grade[PM, MN],
        ),
        (
            "MNRMF",
            "medium naphtha used to produce regular motor fuel",
            pt.naphtha_to_grade[RM, MN],
        ),
        (
            "HNPMF",
            "heavy naphtha used to produce premium motor fuel",
            pt.naphtha_to_grade[PM, HN],
        ),
        (
            "HNRMF",
            "heavy naphtha used to produce regular motor fuel",
            pt.naphtha_to_grade[RM, HN],
        ),
        (
            "RGPMF",
            "reformed gasoline used to produce premium motor fuel",
            pt.reformed_gas_to_grade[PM, RG],
        ),
        (
            "RGRMF",
            "reformed gasoline used to produce regular motor fuel",
            pt.reformed_gas_to_grade[RM, RG],
        ),
        (
            "CGPMF",
            "cracked gasoline used to produce premium motor fuel",
            pt.cracked_gas_to_grade[PM, CG],
        ),
        (
            "CGRMF",
            "cracked gasoline used to produce regular motor fuel",
            pt.cracked_gas_to_grade[RM, CG],
        ),
        # jet fuel
        ("LOJF", "light oil used to produce jet fuel", jf.oil_flow[LO]),
        ("HOJF", "heavy oil used to produce jet fuel", jf.oil_flow[HO]),
        ("RJF", "residuum used to produce jet fuel", jf.residuum_flow),
        ("COJF", "cracked oil used to produce jet fuel", jf.cracked_oil_flow[CO]),
        # lube oil
        ("RLBO", "residuum used to produce lube-oil", lb.residuum_flow),
        # final products
        ("PMF", "premium motor fuel", pt.production[PM]),
        ("RMF", "regular motor fuel", pt.production[RM]),
        ("JF", "jet fuel", jf.production),
        # not in your list, but worth seeing
        ("LOFO", "light oil used to produce fuel oil", fo.oil_flow[LO]),
        ("HOFO", "heavy oil used to produce fuel oil", fo.oil_flow[HO]),
        ("COFO", "cracked oil used to produce fuel oil", fo.cracked_oil_flow[CO]),
        ("RFO", "residuum used to produce fuel oil", fo.residuum_flow),
        ("FO", "fuel oil", fo.production),
        ("LBO", "lube-oil", lb.production),
    ]

    print(f"\n{'Code':<7}{'Description':<68}{'Barrels/day':>12}")
    print("-" * 87)
    for code, desc, var in rows:
        val = v(var, exception=False)
        val_str = f"{val:12.2f}" if val is not None else f"{'n/a':>12}"
        print(f"{code:<7}{desc:<68}{val_str}")

    print("-" * 87)
    print(f"{'Total profit (pence/day):':<75}{v(model.OBJ):12.2f}")


def main():
    solver = pyo.SolverFactory("highs")
    model = pyo.ConcreteModel(name="refinery_optimization")
    model.crude = pyo.Block(rule=_crude)
    model.distillation = pyo.Block(rule=_distillation)
    model.reforming = pyo.Block(rule=_reforming)
    model.cracking = pyo.Block(rule=_cracking)
    model.petrol = pyo.Block(rule=_petrol)
    model.jet_fuel = pyo.Block(rule=_jet_fuel)
    model.fuel_oil = pyo.Block(rule=_fuel_oil)
    model.lube_oil = pyo.Block(rule=_lube_oil)
    _connect_components(model)
    pyo.TransformationFactory("network.expand_arcs").apply_to(model)
    _set_objective(model)
    solver.solve(model)
    report(model)


if __name__ == "__main__":
    main()
