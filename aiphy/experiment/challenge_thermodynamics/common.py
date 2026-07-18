import numpy as np
from scipy.integrate import odeint

from aiphy.experiment import (
    Concept,
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_t,
    default_parastructure,
)


concept_temp = Concept("(1->Particle) |- temp[1]")


def ranged_parastructure(low: float, high: float):
    return default_parastructure(low, high)


def fixed_parastructure(value: float):
    return default_parastructure(value, value)


def make_thermal_obj_info(capacity_ranges):
    obj_info = {}
    for idx, (low, high) in enumerate(capacity_ranges, start=1):
        obj_info[f"o{idx}"] = Objstructure.make_particle(low, high)
    obj_info["clock"] = Objstructure.clock()
    return obj_info


def make_thermal_data_info(num_bodies: int):
    data_info = []
    for idx in range(1, num_bodies + 1):
        data_info.append((concept_temp, [f"o{idx}"]))
    data_info.append((concept_t, ["clock"]))
    return data_info


def make_thermal_expconfig(
    exp_name: str,
    exp_para: dict,
    capacity_ranges,
):
    return ExpConfig(
        exp_name,
        1,
        exp_para,
        make_thermal_obj_info(capacity_ranges),
        make_thermal_data_info(len(capacity_ranges)),
    )


def register_geometry_specs(expconfig: ExpConfig, geometry_specs):
    for geom_type, expr in geometry_specs:
        if geom_type == "zero":
            expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(expr)))
        elif geom_type == "conserved":
            expconfig.register_geometry_info(
                Proposition.IsConserved(expconfig.gen_exp(expr))
            )
        else:
            raise ValueError(f"Unsupported geometry spec: {geom_type}")


def finalize_structure(expconfig: ExpConfig, file_path: str, geometry_specs=None):
    register_geometry_specs(expconfig, geometry_specs or [])
    return ExpStructure(expconfig, DoExpType(file_path))


def time_grid(t_end: float, t_num: int):
    return np.linspace(0.0, t_end, t_num, endpoint=False)


def insert_temperature_data(exp_config: ExpConfig, t, error: float, temp_series):
    data_struct = exp_config.new_datastruct_of_doexperiment(len(t))
    for idx, values in enumerate(temp_series, start=1):
        data_struct.insert_data(
            (concept_temp, [f"o{idx}"]),
            values + np.random.normal(0, error, len(t)),
        )
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, len(t)))
    return data_struct


def integrate_thermal_system(t, y0, rhs):
    sol = odeint(rhs, y0, t)
    if sol.ndim == 1:
        sol = sol[:, None]
    return [sol[:, idx] for idx in range(sol.shape[1])]


def smooth_heat_capacity(temp, c_base, latent_strength, transition_temp, transition_width):
    scaled = (temp - transition_temp) / max(transition_width, 1e-6)
    return c_base + latent_strength * np.exp(-(scaled ** 2))
