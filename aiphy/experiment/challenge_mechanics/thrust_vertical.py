"""
Vertical thrust rocket with observable mass — STRICT EDITION.

A rocket moves vertically (posz) under gravity + time-varying thrust.
Fuel mass is OBSERVABLE. This experiment directly falsifies the simple
"az = const" (R_05) law since az changes due to changing mass.

Observables: posz, mass, t
Physics:
  m(t) = m_payload + m_fuel(t)
  az(t) = (thrust - m(t)*g) / m(t) = thrust/m(t) - g
  thrust = mdot * exhaust_v
After fuel exhaustion: az = -g (free fall)

Key discovery targets:
  - az is NOT constant → falsifies R_05
  - momentum m*v is NOT conserved (variable mass)
  - challenge: integrate rocket equation to test m*dv/dt = thrust - m*g
"""
import numpy as np
from aiphy.experiment import (
    DoExpType, ExpConfig, ExpStructure, Objstructure, Proposition,
    concept_posz, concept_mass, concept_t,
    default_parastructure,
)

exp_para = {
    "mdot": default_parastructure(0.12, 0.35),
    "exhaust_v": default_parastructure(25.0, 45.0),
}

obj_info = {
    "o1": Objstructure.make_particle(1.5, 4.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posz, ["o1"]),
    (concept_mass, ["o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    mdot = exp_config.para("mdot")
    exhaust_v = exp_config.para("exhaust_v")
    payload_mass = exp_config.get_obj_para("o1", "m")
    fuel_mass = payload_mass * 0.7
    g = 9.801234567
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    thrust = mdot * exhaust_v
    burn_time = min(fuel_mass / mdot, t_end)
    a = np.zeros(t_num); v = np.zeros(t_num); posz = np.zeros(t_num)
    mass_t = np.zeros(t_num)
    posz[0] = 5.0; v[0] = -2.0
    for i in range(t_num):
        ct = i * step
        if ct < burn_time:
            m = payload_mass + fuel_mass - mdot * ct
            a[i] = (thrust - m * g) / m
        else:
            m = payload_mass; a[i] = -g
        mass_t[i] = m
    for i in range(1, t_num):
        dt = step
        v[i] = v[i-1] + a[i-1] * dt
        posz[i] = posz[i-1] + v[i-1] * dt + 0.5 * a[i-1] * dt**2
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    ds.insert_data((concept_mass, ["o1"]), mass_t + np.random.normal(0, error * 0.01, t_num))
    ds.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return ds

def thrust_vertical_config():
    expconfig = ExpConfig("thrust_vertical", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)
