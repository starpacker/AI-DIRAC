"""RLC circuit experiment — Reviewer Section 6, Stage 4, experiment 15.

L*q'' + R*q' + (1/C)*q = 0  (series RLC)
Tests second-order linear systems, under/over/critical damping discrimination.
"""
import numpy as np
from scipy.integrate import odeint
from aiphy.experiment import (
    Concept, default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition,
    concept_t,
)
from aiphy.experiment.challenge_thermodynamics.common import ranged_parastructure

concept_charge = Concept("(1->Particle) |- charge[1]")
concept_current = Concept("(1->Particle) |- current[1]")

exp_para = {
    "q0": default_parastructure(0.5, 5.0),
    "i0": default_parastructure(-2.0, 2.0),
    "R": default_parastructure(0.2, 8.0),
    "L": default_parastructure(0.5, 4.0),
    "C": default_parastructure(0.1, 2.0),
}
obj_info = {
    "o1": Objstructure.make_particle(0.1, 5.0),
    "clock": Objstructure.clock(),
}
data_info = [
    (concept_charge, ["o1"]), (concept_current, ["o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    q0 = exp_config.para("q0"); i0 = exp_config.para("i0")
    R = exp_config.para("R"); L = exp_config.para("L"); C = exp_config.para("C")
    step = t_end / t_num; t = np.arange(0, t_end, step)
    def rhs(y, t):
        q, i = y
        return [i, -(R/L)*i - (1/(L*C))*q]
    sol = odeint(rhs, [q0, i0], t)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_charge, ['o1']), sol[:, 0] + np.random.normal(0, error, t_num))
    ds.insert_data((concept_current, ['o1']), sol[:, 1] + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def rlc_circuit_config():
    expconfig = ExpConfig("rlc_circuit", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)
