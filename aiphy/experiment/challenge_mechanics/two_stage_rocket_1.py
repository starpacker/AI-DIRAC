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

exp_para = {
    "z0": default_parastructure(-2.0, 2.0),
    "vz0": default_parastructure(0.0, 2.0),
    "fuel1": default_parastructure(2.0, 4.5),
    "fuel2": default_parastructure(1.5, 3.5),
    "stage1_dry": default_parastructure(1.0, 2.0),
    "stage2_dry": default_parastructure(0.8, 1.6),
    "mdot1": default_parastructure(0.5, 1.2),
    "mdot2": default_parastructure(0.3, 0.9),
    "u1": default_parastructure(18.0, 35.0),
    "u2": default_parastructure(12.0, 28.0),
}

obj_info = {
    "o1": Objstructure.make_particle(2.0, 4.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z0 = exp_config.para("z0")
    vz0 = exp_config.para("vz0")
    fuel1 = exp_config.para("fuel1")
    fuel2 = exp_config.para("fuel2")
    stage1_dry = exp_config.para("stage1_dry")
    stage2_dry = exp_config.para("stage2_dry")
    mdot1 = exp_config.para("mdot1")
    mdot2 = exp_config.para("mdot2")
    u1 = exp_config.para("u1")
    u2 = exp_config.para("u2")
    payload_mass = exp_config.get_obj_para("o1", "m")
    g = 9.801234567

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    burn1_time = fuel1 / mdot1
    burn2_time = fuel2 / mdot2
    t1 = min(t_end, burn1_time)
    t2 = min(t_end, burn1_time + burn2_time)

    def integrate_segment(times, y_init, stage):
        if len(times) == 0:
            return np.empty((0, len(y_init)))

        def f(y, time_value):
            z, v = y
            if stage == 1:
                fuel_mass1 = max(fuel1 - mdot1 * time_value, 0.0)
                total_mass = payload_mass + stage2_dry + fuel2 + stage1_dry + fuel_mass1
                thrust = u1 * mdot1
            elif stage == 2:
                local_time = time_value - burn1_time
                fuel_mass2 = max(fuel2 - mdot2 * local_time, 0.0)
                total_mass = payload_mass + stage2_dry + fuel_mass2
                thrust = u2 * mdot2
            else:
                total_mass = payload_mass + stage2_dry
                thrust = 0.0
            a = thrust / total_mass - g
            return [v, a]

        return odeint(f, y_init, times)

    mask1 = t <= t1 + 1e-12
    seg1_t = t[mask1]
    sol1 = integrate_segment(seg1_t, [z0, vz0], stage=1)

    mask2 = (t > t1 + 1e-12) & (t <= t2 + 1e-12)
    seg2_t = t[mask2]
    if len(seg2_t) > 0:
        seg2_times = np.concatenate(([t1], seg2_t))
        sol2_full = integrate_segment(seg2_times, sol1[-1], stage=2)
        sol2 = sol2_full[1:]
        stage2_start = sol2_full[-1]
    else:
        sol2 = np.empty((0, 2))
        stage2_start = sol1[-1]

    mask3 = t > t2 + 1e-12
    seg3_t = t[mask3]
    if len(seg3_t) > 0:
        seg3_times = np.concatenate(([t2], seg3_t))
        sol3_full = integrate_segment(seg3_times, stage2_start, stage=3)
        sol3 = sol3_full[1:]
    else:
        sol3 = np.empty((0, 2))

    posz = np.concatenate((sol1[:, 0], sol2[:, 0], sol3[:, 0]))

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def two_stage_rocket_1_config() -> ExpStructure:
    expconfig = ExpConfig("two_stage_rocket_1", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posx[o1]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )

    return ExpStructure(expconfig, doexp)
