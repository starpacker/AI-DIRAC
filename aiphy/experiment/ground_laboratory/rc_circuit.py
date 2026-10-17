"""RC circuit experiment — Reviewer Section 6, Stage 4, experiment 14.

V(t) = V_inf + (V0 - V_inf)*exp(-t/RC)
Tests cross-domain structural discovery: same exponential form as Newton cooling,
but with T^2*I^-2*L^-2 dimension instead of temperature.
"""
import numpy as np
from aiphy.experiment import (
    Concept, default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition,
    concept_t,
)
concept_voltage = Concept("(1->Particle) |- voltage[1]")

exp_para = {
    "V0": default_parastructure(6.0, 18.0),
    "V_inf": default_parastructure(0.0, 3.0),
    "R": default_parastructure(0.5, 5.0),
    "C": default_parastructure(0.1, 2.0),
}
obj_info = {
    "o1": Objstructure.make_particle(0.1, 5.0),
    "clock": Objstructure.clock(),
}
data_info = [
    (concept_voltage, ["o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    V0 = exp_config.para("V0"); V_inf = exp_config.para("V_inf")
    R = exp_config.para("R"); C = exp_config.para("C")
    tau = R * C
    step = t_end / t_num; t = np.arange(0, t_end, step)
    V = V_inf + (V0 - V_inf) * np.exp(-t / tau)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_voltage, ['o1']), V + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def rc_circuit_config():
    expconfig = ExpConfig("rc_circuit", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)
