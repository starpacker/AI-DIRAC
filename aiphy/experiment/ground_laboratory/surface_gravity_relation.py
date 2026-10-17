import numpy as np

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


G = 6.6e-11
SURFACE_G = 9.801234567
EARTH_RADIUS = 1000.0
EARTH_MASS = SURFACE_G * EARTH_RADIUS ** 2 / G

exp_para = {
    "z0": default_parastructure(0.0, 2.0),
}

obj_info = {
    "earth": Objstructure.make_particle(0.65 * EARTH_MASS, 1.35 * EARTH_MASS),
    "probe": Objstructure.make_particle(1.0, 2.0),
    "radius": Objstructure.make_spring(1.5, 1.5, 0.75 * EARTH_RADIUS, 1.35 * EARTH_RADIUS),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_length, ["radius"]),
    (concept_posx, ["probe"]),
    (concept_posy, ["probe"]),
    (concept_posz, ["probe"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z0 = exp_config.para("z0")
    central_mass = exp_config.get_obj_para("earth", "m")
    radius = exp_config.get_obj_para("radius", "freel")
    g = G * central_mass / (radius ** 2)

    step = t_end / t_num
    t = np.arange(0, t_end, step)
    z = z0 - 0.5 * g * t ** 2

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_length, ["radius"]), np.full(t_num, radius) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ["probe"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["probe"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["probe"]), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def surface_gravity_relation_config() -> ExpStructure:
    expconfig = ExpConfig(
        "surface_gravity_relation",
        1,
        exp_para,
        obj_info,
        data_info,
    )
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posx[probe]"))
    )
    expconfig.register_geometry_info(
        Proposition.IsZero(expconfig.gen_exp("posy[probe]"))
    )

    return ExpStructure(expconfig, doexp)
