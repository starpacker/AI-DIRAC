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

# Two particles on frictionless 1D track with elastic collision.
# Particles approach, collide, and bounce — momentum and KE are conserved.
# v1 and v2 are velocities in lab frame BEFORE collision.
# The collision happens exactly mid-experiment.

exp_para = {
    "v1": default_parastructure(2.0, 5.0),
    "v2": default_parastructure(-3.0, -1.0),
    "sep": default_parastructure(2.0, 5.0),
}

obj_info = {
    "o1": Objstructure.make_particle(1.0, 3.0),
    "o2": Objstructure.make_particle(2.0, 5.0),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o1"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o1"]),
    (concept_posz, ["o2"]),
    (concept_mass, ["o1"]),
    (concept_mass, ["o2"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    v1 = exp_config.para("v1")
    v2 = exp_config.para("v2")
    sep = exp_config.para("sep")

    m1 = exp_config.get_obj_para("o1", "m")
    m2 = exp_config.get_obj_para("o2", "m")

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    # Collision time: initially separated by sep, approach at v1-v2
    # Particle 1 starts at x=0, particle 2 at x=sep
    # Collision when x1 = x2: v1*tc = sep + v2*tc → tc = sep/(v1-v2)
    tc = sep / (v1 - v2)
    t_mid = t_end / 2
    # Shift so collision happens at t=t_mid
    start_time = t_mid - tc

    # Post-collision velocities (elastic)
    tot_m = m1 + m2
    v1_post = ((m1 - m2) * v1 + 2 * m2 * v2) / tot_m
    v2_post = ((m2 - m1) * v2 + 2 * m1 * v1) / tot_m

    x1 = np.zeros(t_num)
    x2 = np.zeros(t_num)

    for i, ti in enumerate(t):
        if ti < t_mid:
            x1[i] = v1 * (ti + start_time)
            x2[i] = sep + v2 * (ti + start_time)
        else:
            dt = ti - t_mid
            x1_c = v1 * tc  # position at collision
            x1[i] = x1_c + v1_post * dt
            x2[i] = x1_c + v2_post * dt

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ["o2"]), x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o2"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o2"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    # Emit masses as constant columns — intrinsic object properties needed
    # for discovering momentum (m₁v₁ + m₂v₂ = const) and kinetic energy
    # (½m₁v₁² + ½m₂v₂² = const) in elastic collisions.
    m1 = exp_config.get_obj_para("o1", "m")
    m2 = exp_config.get_obj_para("o2", "m")
    data_struct.insert_data((concept_mass, ["o1"]), np.full(t_num, m1))
    data_struct.insert_data((concept_mass, ["o2"]), np.full(t_num, m2))
    return data_struct


def collision_1d_elastic_config() -> ExpStructure:
    expconfig = ExpConfig("collision_1d_elastic", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posy[o1]"))
    )
    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posy[o2]"))
    )
    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posz[o1]"))
    )
    expconfig.register_geometry_info(
    Proposition.IsZero(expconfig.gen_exp("posz[o2]"))
    )

    return ExpStructure(expconfig, doexp)
