import numpy as np
from scipy.integrate import odeint

from aiphy.experiment import (
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_dist,
    concept_length,
    concept_posx,
    concept_posy,
    concept_posz,
    concept_t,
    default_parastructure,
)

exp_para = {
    "z0": default_parastructure(10.0, 12.0),
    "z10": default_parastructure(-5.0, 5.0),
    "z20": default_parastructure(-15.0, -10.0),
    "vz10": default_parastructure(-1.0, 1.0),
    "vz20": default_parastructure(-1.0, 1.0),
    "c1": default_parastructure(0.2, 0.8),
    "c2": default_parastructure(0.2, 0.8),
    "f0": default_parastructure(4.0, 16.0),
    "omega": default_parastructure(0.5, 1.8),
}

obj_info = {
    "o1": Objstructure.make_particle(5.0, 6.0),
    "s1": Objstructure.make_spring(1.7, 1.9, 5.0, 10.0),
    "o2": Objstructure.make_particle(5.0, 6.0),
    "s2": Objstructure.make_spring(1.7, 1.9, 5.0, 10.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_length, ["s2"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z0 = exp_config.para("z0")
    z10 = exp_config.para("z10")
    z20 = exp_config.para("z20")
    vz10 = exp_config.para("vz10")
    vz20 = exp_config.para("vz20")
    c1 = exp_config.para("c1")
    c2 = exp_config.para("c2")
    f0 = exp_config.para("f0")
    omega = exp_config.para("omega")
    l1 = exp_config.get_obj_para("s1", "freel")
    k1 = exp_config.get_obj_para("s1", "thickness") ** 3
    m1 = exp_config.get_obj_para("o1", "m")
    l2 = exp_config.get_obj_para("s2", "freel")
    k2 = exp_config.get_obj_para("s2", "thickness") ** 3
    m2 = exp_config.get_obj_para("o2", "m")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    def f(y, time_value):
        z1, z2, v1, v2 = y
        drive = f0 * np.cos(omega * time_value)
        a1 = -g + (
            k1 * (z0 - z1 - l1) - c1 * v1 - k2 * (z1 - z2 - l2) - c2 * (v1 - v2) + drive
        ) / m1
        a2 = -g + (k2 * (z1 - z2 - l2) + c2 * (v1 - v2)) / m2
        return [v1, v2, a1, a2]

    sol = odeint(f, [z10, z20, vz10, vz20], t)
    posz1 = sol[:, 0]
    posz2 = sol[:, 1]
    length1 = z0 - posz1
    length2 = posz1 - posz2

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ["o2"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o2"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o2"]), posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s1"]), length1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s2"]), length2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o1", "o2"]), length2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o2", "o1"]), length2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct

def driven_damped_hanging_spring_2_config() -> ExpStructure:
    expconfig = ExpConfig("driven_damped_hanging_spring_2", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posx[o1]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posx[o2]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[o2]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("dist[o1, o2] - dist[o2, o1]"))
    )
    return ExpStructure(expconfig, doexp)
