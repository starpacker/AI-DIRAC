import numpy as np
from scipy.integrate import odeint

from aiphy.experiment import (
    DoExpType, ExpConfig, ExpStructure, Objstructure, Proposition,
    concept_posx, concept_posy, concept_posz, concept_length, concept_mass, concept_t,
    default_parastructure,
)

# Undamped simple pendulum — conservative system.
# Energy: ½θ̇² + (g/L)(1-cos θ) = const
# Angular momentum: m·L²·θ̇ = const
# Small-angle: posx ≈ L·θ, posz ≈ -L·(1-θ²/2)
# This is the ideal experiment for energy/angular-momentum discovery.

exp_para = {
    "theta0": default_parastructure(-0.8, 0.8),
    "omega0": default_parastructure(-0.5, 0.5),
}

obj_info = {
    "o1": Objstructure.make_particle(1.0, 3.0),
    "s1": Objstructure.make_spring(1.5, 2.0, 2.0, 3.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_mass, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    theta0 = exp_config.para("theta0")
    omega0 = exp_config.para("omega0")
    length = exp_config.get_obj_para("s1", "freel")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    def f(y, time_value):
        theta, omega = y
        alpha = -(g / length) * np.sin(theta)
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
    m = exp_config.get_obj_para("o1", "m")
    data_struct.insert_data((concept_mass, ["o1"]), np.full(t_num, m))
    return data_struct


def pendulum_undamped_1_config() -> ExpStructure:
    expconfig = ExpConfig("pendulum_undamped_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
