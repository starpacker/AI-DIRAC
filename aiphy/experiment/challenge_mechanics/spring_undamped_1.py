import numpy as np
from scipy.integrate import odeint

from aiphy.experiment import (
    DoExpType, ExpConfig, ExpStructure, Objstructure, Proposition,
    concept_posx, concept_posy, concept_posz, concept_length, concept_mass, concept_t,
    default_parastructure,
)

# Undamped mass-spring oscillator — conservative system.
# Force: m·d²z/dt² = -k·(z - z_eq)
# Energy: ½m·v² + ½k·(z-z_eq)² = const
# Frequency: ω₀² = k/m

exp_para = {
    "z0": default_parastructure(-4.0, 2.0),
    "vz0": default_parastructure(-2.0, 2.0),
}

obj_info = {
    "o1": Objstructure.make_particle(1.0, 3.0),
    "s1": Objstructure.make_spring(1.7, 2.2, 1.5, 2.5),
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
    zl = 10.0  # ceiling height, fixed
    z0 = exp_config.para("z0")
    vz0 = exp_config.para("vz0")
    l0 = exp_config.get_obj_para("s1", "freel")  # natural length
    k = exp_config.get_obj_para("s1", "thickness") ** 3
    m = exp_config.get_obj_para("o1", "m")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    # Equilibrium position: k*(z_eq - zl + l0) = m*g
    # So z_eq = zl - l0 - m*g/k
    z_eq = zl - l0 - m * g / k
    r0 = zl - z0  # initial spring length

    def f(y, time_value):
        z, v = y
        r = zl - z  # spring extension
        a = g - k * (r - l0) / m
        return [v, a]

    sol = odeint(f, [z0, vz0], t)
    posz = sol[:, 0]
    length = zl - posz  # spring length

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s1"]), length + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    m = exp_config.get_obj_para("o1", "m")
    data_struct.insert_data((concept_mass, ["o1"]), np.full(t_num, m))
    return data_struct


def spring_undamped_1_config() -> ExpStructure:
    expconfig = ExpConfig("spring_undamped_1", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posx[o1]"))
    )
    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
