"""
Horizontal spring-mass oscillator — STRICT EDITION.

A mass on a spring oscillates horizontally (posx only), with no gravity
interference. ax ≠ 0 and is a function of position.
This is the cleanest test for the full energy conservation law.

Observables: posx, t
Hidden: spring_constant k, mass m
Physics: a = -k*x/m
Energy: 0.5*m*v^2 + 0.5*k*x^2 = const

Key discovery targets:
  - ax is NOT constant → falsifies ax=0 laws
  - v^2 + (k/m)*x^2 = const → mechanical energy conservation
  - Generalization of v^2 - 2g*z beyond free fall
"""
import numpy as np
from aiphy.experiment import (
    DoExpType, ExpConfig, ExpStructure, Objstructure, Proposition,
    concept_posx, concept_t,
    default_parastructure,
)

exp_para = {
    "k": default_parastructure(3.0, 8.0),
    "x0": default_parastructure(0.5, 2.0),
}

obj_info = {
    "o1": Objstructure.make_particle(0.8, 2.5),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    k = exp_config.para("k")
    x0 = exp_config.para("x0")
    m = exp_config.get_obj_para("o1", "m")
    omega = np.sqrt(k / m)
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    posx = x0 * np.cos(omega * t)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return ds

def spring_horizontal_config():
    expconfig = ExpConfig("spring_horizontal", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)
