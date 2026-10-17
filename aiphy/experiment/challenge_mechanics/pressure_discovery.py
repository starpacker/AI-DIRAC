"""Pressure discovery experiment — STRICT EDITION.

A piston of known mass rests on a gas column.  The piston moves vertically.
Observables: piston position (posz), piston mass, piston cross-section area,
and the spring constant (if the piston is spring-loaded) are NOT directly
observable — only posz, mass, and cross-section dimensions.

The gas undergoes slow isothermal expansion/compression driven by an
external heater/cooler.  The piston rises/falls, giving time-varying
position data.

From posz(t), the engine can derive:
  - velocity v = D[posz]/D[t]
  - acceleration a = D[v]/D[t]
  - net force on piston: F_net = mass * a
  - gas pressure: P = (mass*g + mass*a) / area  (if g is external field)

Key discovery target: the force per unit area exerted by the gas on the
piston bottom face is conserved when the process is quasi-static and
isobaric — this is the *definition* of pressure.

Observables (5): posx, posy, posz, mass, t
Geometry: the piston only moves vertically (posx=posy=0)
"""

import numpy as np

from aiphy.experiment import (
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_posx,
    concept_posy,
    concept_posz,
    concept_mass,
    concept_t,
    default_parastructure,
)

exp_para = {
    "area": default_parastructure(0.2, 0.5),        # piston area (hidden)
    "gas_pressure": default_parastructure(1.0, 3.0), # nominal gas pressure (hidden)
}

obj_info = {
    "piston": Objstructure.make_particle(0.5, 2.0),  # piston mass
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["piston"]),
    (concept_posy, ["piston"]),
    (concept_posz, ["piston"]),
    (concept_mass, ["piston"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    area = exp_config.para("area")
    P_nominal = exp_config.para("gas_pressure")
    m = exp_config.get_obj_para("piston", "m")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    # The gas pressure varies slowly (quasi-static process).
    # The piston accelerometer force balance: m*a = P*area - m*g
    # So a = (P*area - m*g) / m
    # The piston moves accordingly.

    # Slow pressure variation
    P_t = P_nominal * (1.0 + 0.15 * np.sin(2.0 * np.pi * t / t_end * 1.3)
                       + 0.08 * np.sin(2.0 * np.pi * t / t_end * 2.1))

    # Force from gas: F_gas = P_t * area
    # Net force: F_net = P_t * area - m*g
    # Acceleration: a = F_net / m
    a_t = (P_t * area - m * g) / m

    # Integrate numerically to get velocity and position
    v = np.zeros(t_num)
    posz = np.zeros(t_num)
    posz[0] = 0.5 + 0.1 * np.sin(0.7)  # random-ish starting position

    for i in range(1, t_num):
        v[i] = v[i-1] + a_t[i-1] * step
        posz[i] = posz[i-1] + v[i-1] * step + 0.5 * a_t[i-1] * step**2

    # Add measurement noise
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["piston"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["piston"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["piston"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_mass, ["piston"]), np.full(t_num, m))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def pressure_discovery_config() -> ExpStructure:
    expconfig = ExpConfig("pressure_discovery", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    # Geometry: piston only moves vertically
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posx[piston]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[piston]"))
    )

    return ExpStructure(expconfig, doexp)
