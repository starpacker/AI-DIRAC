"""
Horizontal thrust rocket experiment — STRICT EDITION.

A point mass moves horizontally (posx) under time-varying thrust.
No gravity — pure horizontal dynamics.
Fuel mass is OBSERVABLE (decreases over time).
The engine can discover F = m*a even when mass changes.

Observables: posx, posy, posz, mass, t
"""
import numpy as np
from aiphy.experiment import (
    DoExpType, ExpConfig, ExpStructure, Objstructure, Proposition,
    concept_posx, concept_posy, concept_posz, concept_mass, concept_t,
    default_parastructure,
)

exp_para = {
    "mdot": default_parastructure(0.15, 0.45),
    "exhaust_v": default_parastructure(20.0, 40.0),
}

obj_info = {
    "o1": Objstructure.make_particle(2.0, 5.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_mass, ["o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    mdot = exp_config.para("mdot")
    exhaust_v = exp_config.para("exhaust_v")
    payload_mass = exp_config.get_obj_para("o1", "m")
    fuel_mass = payload_mass * 0.8
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    thrust = mdot * exhaust_v
    burn_time = min(fuel_mass / mdot, t_end)
    a = np.zeros(t_num); v = np.zeros(t_num); posx = np.zeros(t_num)
    mass_t = np.zeros(t_num)
    posx[0] = 0.5; v[0] = 1.0
    for i in range(t_num):
        ct = i * step
        if ct < burn_time:
            m = payload_mass + fuel_mass - mdot * ct
            a[i] = thrust / m
        else:
            m = payload_mass; a[i] = 0.0
        mass_t[i] = m
    for i in range(1, t_num):
        dt = step
        v[i] = v[i-1] + a[i-1] * dt
        posx[i] = posx[i-1] + v[i-1] * dt + 0.5 * a[i-1] * dt**2
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    ds.insert_data((concept_posz, ["o1"]), np.random.normal(0, error, t_num))
    ds.insert_data((concept_mass, ["o1"]), mass_t + np.random.normal(0, error * 0.01, t_num))
    ds.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return ds

def thrust_horizontal_config():
    expconfig = ExpConfig("thrust_horizontal", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posy[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posz[o1]")))
    return ExpStructure(expconfig, doexp)
