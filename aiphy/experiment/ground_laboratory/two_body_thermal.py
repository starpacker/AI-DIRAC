"""Two-body thermal exchange — Reviewer Section 6, thermodynamics experiment 13.

dT1/dt = -k1*(T1 - T2)
dT2/dt = k2*(T1 - T2) - k3*(T2 - Tenv)
Tests multi-variable dynamics, multi-exponential response, and coupling detection.
"""
import numpy as np
from scipy.integrate import odeint
from aiphy.experiment import (
    Concept, ExpConfig, ExpStructure, Objstructure, DoExpType, Proposition,
    concept_t, default_parastructure,
)

concept_temp = Concept("(1->Particle) |- temp[1]")

exp_para = {
    "T1_0": default_parastructure(60.0, 120.0),
    "T2_0": default_parastructure(10.0, 40.0),
    "Tenv": default_parastructure(0.0, 15.0),
    "k1": default_parastructure(0.1, 0.5),
    "k2": default_parastructure(0.1, 0.5),
    "k3": default_parastructure(0.05, 0.3),
}
obj_info = {
    "o1": Objstructure.make_particle(0.5, 2.0),
    "o2": Objstructure.make_particle(0.5, 2.0),
    "clock": Objstructure.clock(),
}
data_info = [
    (concept_temp, ["o1"]), (concept_temp, ["o2"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    T1_0 = exp_config.para("T1_0"); T2_0 = exp_config.para("T2_0")
    Tenv = exp_config.para("Tenv")
    k1 = exp_config.para("k1"); k2 = exp_config.para("k2"); k3 = exp_config.para("k3")
    c1 = exp_config.get_obj_para("o1", "m"); c2 = exp_config.get_obj_para("o2", "m")
    step = t_end / t_num; t = np.arange(0, t_end, step)
    def rhs(y, t):
        T1, T2 = y
        return [
            -(k1/c1)*(T1 - T2),
            (k2/c2)*(T1 - T2) - (k3/c2)*(T2 - Tenv),
        ]
    sol = odeint(rhs, [T1_0, T2_0], t)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_temp, ['o1']), sol[:, 0] + np.random.normal(0, error, t_num))
    ds.insert_data((concept_temp, ['o2']), sol[:, 1] + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def two_body_thermal_config():
    expconfig = ExpConfig("two_body_thermal", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)
