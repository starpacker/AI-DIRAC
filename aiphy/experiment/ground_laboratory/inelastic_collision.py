"""Inelastic collision experiments — Reviewer Section 6, collision_y Variant C.

Two particles collide with restitution coefficient 0 < e < 1.
Tests that the system does NOT confuse geometric identities with conservation laws.
"""
import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist

exp_para = {
    "y1": default_parastructure(-5.0, -3.0),
    "v1": default_parastructure(2.0, 6.0),
    "y2": default_parastructure(3.0, 5.0),
    "v2": default_parastructure(-6.0, -2.0),
    "restitution": default_parastructure(0.2, 0.8),
}
obj_info = {
    "MPa": Objstructure.make_particle(2.0, 10.0),
    "MPb": Objstructure.make_particle(2.0, 10.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["MPa"]), (concept_posx, ["MPb"]),
    (concept_posy, ["MPa"]), (concept_posy, ["MPb"]),
    (concept_posz, ["MPa"]), (concept_posz, ["MPb"]),
    (concept_dist, ["MPa", "MPb"]), (concept_dist, ["MPb", "MPa"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    y1 = exp_config.para("y1"); y2 = exp_config.para("y2")
    v1 = exp_config.para("v1"); v2 = exp_config.para("v2")
    e = exp_config.para("restitution")
    m1 = exp_config.get_obj_para("MPa", "m"); m2 = exp_config.get_obj_para("MPb", "m")
    step = t_end / t_num; t = np.arange(0, t_end, step)
    t_col = (y2 - y1) / (v1 - v2)
    assert t_col > 0 and t_col < t_end
    vcm = (m1*v1 + m2*v2) / (m1 + m2)
    vr = v1 - v2
    vn1 = vcm + m2/(m1+m2) * (-e*vr)
    vn2 = vcm - m1/(m1+m2) * (-e*vr)
    data_y1 = np.where(t < t_col, y1 + v1*t, y1 + v1*t_col + vn1*(t-t_col))
    data_y2 = np.where(t < t_col, y2 + v2*t, y2 + v2*t_col + vn2*(t-t_col))
    data_dist = np.abs(data_y2 - data_y1)
    zeros = np.zeros(t_num)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    for obj, vals in [("MPa", data_y1), ("MPb", data_y2)]:
        ds.insert_data((concept_posx, [obj]), zeros + np.random.normal(0, error, t_num))
        ds.insert_data((concept_posy, [obj]), vals + np.random.normal(0, error, t_num))
        ds.insert_data((concept_posz, [obj]), zeros + np.random.normal(0, error, t_num))
    ds.insert_data((concept_dist, ["MPa", "MPb"]), data_dist + np.random.normal(0, error, t_num))
    ds.insert_data((concept_dist, ["MPb", "MPa"]), data_dist + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return ds

def inelastic_collision_config():
    expconfig = ExpConfig("inelastic_collision", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("dist[MPa, MPb] - dist[MPb, MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posx[MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posx[MPb]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posz[MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posz[MPb]")))
    return ExpStructure(expconfig, doexp)
