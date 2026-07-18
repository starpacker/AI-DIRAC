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

pi = 3.1415926

exp_para = {
    "theta0": default_parastructure(-pi / 2.5, pi / 2.5),
    "omega0": default_parastructure(-1.5, 1.5),
    "c": default_parastructure(0.1, 0.8),
}

obj_info = {
    "o1": Objstructure.make_particle(3.0, 6.0),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
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
    theta0 = exp_config.para("theta0")
    omega0 = exp_config.para("omega0")
    c = exp_config.para("c")
    length = exp_config.get_obj_para("s1", "freel")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    def f(y, time_value):
        theta, omega = y
        alpha = -(g / length) * np.sin(theta) - c * omega
        return [omega, alpha]

    sol = odeint(f, [theta0, omega0], t)
    theta = sol[:, 0]
    posx = length * np.sin(theta)
    posz = -length * np.cos(theta)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s1"]), np.full(t_num, length) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def damped_pendulum_1_config() -> ExpStructure:
    expconfig = ExpConfig("damped_pendulum_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posz[o1]**2 + posx[o1]**2 - length[s1]**2"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
