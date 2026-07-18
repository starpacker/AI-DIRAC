import numpy as np
from scipy.integrate import odeint

from aiphy.experiment import (
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_length,
    concept_posx,
    concept_posy,
    concept_posz,
    concept_t,
    default_parastructure,
)

exp_para = {
    "zl": default_parastructure(10.0, 12.0),
    "zr0": default_parastructure(-6.0, 2.0),
    "vz0": default_parastructure(-1.5, 1.5),
    "c": default_parastructure(0.3, 1.2),
}

obj_info = {
    "o1": Objstructure.make_particle(5.0, 6.0),
    "s1": Objstructure.make_spring(1.7, 1.9, 13.0, 17.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    zl = exp_config.para("zl")
    zr0 = exp_config.para("zr0")
    vz0 = exp_config.para("vz0")
    c = exp_config.para("c")
    l = exp_config.get_obj_para("s1", "freel")
    k = exp_config.get_obj_para("s1", "thickness") ** 3
    m = exp_config.get_obj_para("o1", "m")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)
    r0 = zl - zr0

    def f(y, time_value):
        r, v = y
        a = g - k * (r - l) / m - c * v / m
        return [v, a]

    sol = odeint(f, [r0, vz0], t)
    length = sol[:, 0]
    posz = zl - length

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s1"]), length + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def damped_hanging_spring_1_config() -> ExpStructure:
    expconfig = ExpConfig("damped_hanging_spring_1", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
        Proposition.IsConserved(expconfig.gen_exp("posz[o1] + length[s1]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posx[o1]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
