import numpy as np
from scipy.integrate import odeint

from aiphy.experiment import (
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_posx,
    concept_posy,
    concept_posz,
    concept_t,
    default_parastructure,
)

pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-2.0, 2.0),
    "z0": default_parastructure(-2.0, 2.0),
    "vx0": default_parastructure(0.0, 2.0),
    "vz0": default_parastructure(0.0, 3.0),
    "fuel0": default_parastructure(2.0, 5.0),
    "mdot": default_parastructure(0.4, 1.2),
    "u": default_parastructure(15.0, 35.0),
    "theta": default_parastructure(pi / 10, pi / 4),
}

obj_info = {
    "o1": Objstructure.make_particle(3.0, 6.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x0 = exp_config.para("x0")
    z0 = exp_config.para("z0")
    vx0 = exp_config.para("vx0")
    vz0 = exp_config.para("vz0")
    fuel0 = exp_config.para("fuel0")
    mdot = exp_config.para("mdot")
    u = exp_config.para("u")
    theta = exp_config.para("theta")
    payload_mass = exp_config.get_obj_para("o1", "m")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    def f(y, time_value):
        x, z, vx, vz, fuel_mass = y
        burning = fuel_mass > 1e-8
        current_mdot = mdot if burning else 0.0
        total_mass = payload_mass + max(fuel_mass, 0.0)
        thrust = u * current_mdot if burning else 0.0
        ax = thrust * np.sin(theta) / total_mass
        az = thrust * np.cos(theta) / total_mass - g
        return [vx, vz, ax, az, -current_mdot]

    sol = odeint(f, [x0, z0, vx0, vz0, fuel0], t)
    posx = sol[:, 0]
    posz = sol[:, 1]

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def variable_mass_rocket_2_config() -> ExpStructure:
    expconfig = ExpConfig("variable_mass_rocket_2", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
