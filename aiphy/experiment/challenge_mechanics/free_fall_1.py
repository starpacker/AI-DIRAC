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
    concept_mass,
    concept_t,
    default_parastructure,
)

# Free-fall: particle under uniform gravity, NO damping.
# Physics: ddposz/dt = -g (constant absolute acceleration).
# Without initial condition, the engine discovers:
#   - posx=0, posy=0 (coordinates)
#   - dposx_dt=0 (no horizontal motion)
#   - dposz_dt varies → velocity concept registers via cold-start
#   - ddposz_dt = const → acceleration concept via second-derivative bootstrap

exp_para = {
    "z0": default_parastructure(5.0, 10.0),
    "vz0": default_parastructure(-5.0, 5.0),
}

obj_info = {
    "o1": Objstructure.make_particle(1.0, 5.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_mass, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z0 = exp_config.para("z0")
    vz0 = exp_config.para("vz0")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    def f(y, time_value):
        z, v = y
        return [v, -g]

    sol = odeint(f, [z0, vz0], t)
    posz = sol[:, 0]

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    # Emit mass as a constant data column — it is an intrinsic object
    # property that the regression engine needs for discovering momentum
    # (m·v) and kinetic energy (m·v²) conservation.
    m = exp_config.get_obj_para("o1", "m")
    data_struct.insert_data((concept_mass, ["o1"]), np.full(t_num, m))
    return data_struct


def free_fall_1_config() -> ExpStructure:
    expconfig = ExpConfig("free_fall_1", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posx[o1]"))
    )
    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
