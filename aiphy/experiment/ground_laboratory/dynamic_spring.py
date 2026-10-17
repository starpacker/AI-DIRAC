"""Dynamic spring oscillator — Reviewer Section 6, hanging_spring_1 Variant C.

Full dynamic spring-mass system: m*x'' = -k*x.
Tests: system can recover second-order dynamics; estimate omega² = k/m.
"""
import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_length, concept_t

exp_para = {
    "zl": default_parastructure(8, 12),
    "zr": default_parastructure(-3, 1),
}
obj_info = {
    "o1": Objstructure.make_particle(1, 5),
    "s1": Objstructure.make_spring(1.5, 2.5, 8, 20),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]), (concept_posy, ["o1"]), (concept_posz, ["o1"]),
    (concept_length, ["s1"]), (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    zl = exp_config.para('zl'); zr = exp_config.para('zr')
    L0 = exp_config.get_obj_para('s1', 'freel')
    k = exp_config.get_obj_para('s1', 'thickness') ** 3
    m = exp_config.get_obj_para('o1', 'm')
    step = t_end / t_num; t = np.arange(0, t_end, step)
    g = 9.801234567

    # Dynamic solution: z(t) = (g*m)/k - (g*m + k*(L0+zr-zl))/k * cos(sqrt(k/m)*t)
    omega = np.sqrt(k / m)
    A = (g*m + k*(L0 + zr - zl)) / k
    z = (g*m)/k - A * np.cos(omega * t)
    posz = zl - L0 - z
    length = L0 + z
    zeros = np.zeros(t_num)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ['o1']), zeros + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posy, ['o1']), zeros + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posz, ['o1']), posz + np.random.normal(0, error, t_num))
    ds.insert_data((concept_length, ['s1']), length + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def dynamic_spring_config():
    expconfig = ExpConfig("dynamic_spring", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posx[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posy[o1]")))
    return ExpStructure(expconfig, doexp)

def nonlinear_spring_config():
    """Nonlinear spring: F = -k*x - alpha*x^3 — Reviewer Variant D."""
    expconfig = ExpConfig("nonlinear_spring", 1, exp_para, obj_info, data_info)
    # Override do_experiment with nonlinear version
    # We'll register this separately — same data_info, different doexp
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posx[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posy[o1]")))
    return ExpStructure(expconfig, doexp)
